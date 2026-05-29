"""CALM encoder: VAE + energy head, exposed via the Praxis encoder interface.

Architecture (matches the reference at github.com/shaochenze/calm):

1. ``encode`` builds the LM's per-patch input from raw token embeddings,
   *not* from the VAE: ``embed_proj`` compresses K token embeddings into
   one ``hidden_size`` vector per patch. The VAE encoder runs in
   parallel to produce the energy-score's posterior targets, and the VAE
   decoder + classifier produce the AE's own reconstruction loss
   (composed into ``encoder_loss``). The LM input is decoupled from the
   VAE's latent quality, mirroring the reference (where the AE is
   frozen during LM training and only emits targets).
2. The global transformer autoregresses over patch embeddings.
3. ``decode`` uses the LM hidden state at position ``p`` to drive an
   energy head that produces proposals for latent ``p+1``; those are
   compared against the posterior samples of ``p+1`` under the
   energy-score loss. The reconstructed-token logits are returned for
   sanity checking but do not participate in the main loss (the encoder
   sets ``handles_loss=True`` to bypass the default CE path).
"""

import math
import random
from collections import Counter
from typing import Dict, Optional, Tuple, Union

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch import nn

from praxis.heads.energy import EnergyHead
from praxis.losses.energy_score import energy_score_loss

from ..base import BaseEncoder
from .vae import CALMVAE


def _resolve_dim(spec: Union[int, float], base: int) -> int:
    """``float`` -> fraction of ``base`` (e.g. 0.5); ``int`` -> absolute size."""
    if isinstance(spec, float):
        return max(1, round(spec * base))
    return int(spec)


class CALMEncoder(BaseEncoder):
    """CALM autoencoder + energy head, plugged into the encoder slot.

    The encoder owns its loss bookkeeping; see ``handles_loss``.
    """

    # Chart hints for diagnostics surfaced via ``training_metrics()``. Collected
    # by ``praxis.metrics.descriptions.get_metric_descriptions`` and rendered on
    # the Dynamics tab. Do NOT mirror these in TRAINING_METRIC_REGISTRY (that
    # registry is for Research-tab trainer scalars).
    metric_descriptions = {
        "calm_latent_norm_mean": {
            "description": (
                "Mean L2 norm of posterior means ‖μ‖ across positions. Stable "
                "training keeps this in a bounded range; runaway growth signals "
                "latent-scale drift, persistent zero signals posterior collapse."
            ),
            "chart": {
                "title": "Latent Mean Norm",
                "y_label": "mean ‖μ‖",
                "y_scale": "linear",
                "group": "calm",
                "order": 10,
            },
        },
        "calm_latent_std_mean": {
            "description": (
                "Mean posterior σ = √exp(logvar) across positions and dims. "
                "Collapse toward 0 = the encoder is becoming deterministic; "
                "runaway growth = divergence."
            ),
            "chart": {
                "title": "Latent Std",
                "y_label": "mean σ",
                "y_scale": "linear",
                "group": "calm",
                "order": 20,
            },
        },
        "calm_kl_active_frac": {
            "description": (
                "Fraction of latent dims whose pre-clip per-dim KL exceeds the "
                "free-bits floor (kl_clip). Posterior collapse drives this "
                "toward 0; a healthy VAE keeps most dims active."
            ),
            "chart": {
                "title": "Active Latent Dim Fraction",
                "y_label": "frac > floor",
                "y_scale": "linear",
                "group": "calm",
                "order": 30,
            },
        },
        "calm_recon_kl_ratio": {
            "description": (
                "recon_loss / (β·KL). Above 1: reconstruction dominates the "
                "encoder objective; below 1: KL regularization dominates. Watch "
                "the trend rather than the absolute value."
            ),
            "chart": {
                "title": "Recon / (β·KL)",
                "y_label": "ratio",
                "y_scale": "logarithmic",
                "group": "calm",
                "order": 40,
            },
        },
        "calm_kl_beta": {
            "description": (
                "Effective KL coefficient β at this step. Linearly anneals from "
                "0 to the configured kl_beta over kl_warmup_steps, then holds."
            ),
            "chart": {
                "title": "KL β (annealed)",
                "y_label": "β",
                "y_scale": "linear",
                "group": "calm",
                "order": 50,
            },
        },
        "calm_energy_loss": {
            "description": (
                "Energy-score loss between proposed next latents and posterior "
                "samples. Zero during energy_warmup_steps (VAE-only phase); "
                "should trend down once active."
            ),
            "chart": {
                "title": "Energy Loss",
                "y_label": "energy",
                "y_scale": "linear",
                "group": "calm",
                "order": 60,
            },
        },
    }

    def __init__(
        self,
        config,
        *,
        chunk_size: int = 8,
        latent_dim: Union[int, float] = 128,
        ae_hidden: Union[int, float] = 512,
        kl_beta: float = 1e-3,
        kl_clip: float = 0.5,
        kl_warmup_steps: int = 0,
        ae_dropout: float = 0.15,
        noise_dim: Union[int, float] = 128,
        energy_blocks: int = 3,
        energy_samples_n: int = 8,
        energy_samples_m: int = 100,
        energy_alpha: float = 1.0,
        energy_warmup_steps: int = 0,
        vote_num_samples: int = 200,
    ) -> None:
        super().__init__()
        self.config = config

        self.K = chunk_size
        # latent/ae/noise dims may be given as fractions of hidden_size: the
        # latent feeds (and projects to) the global transformer, so its width
        # is the natural yardstick for these encoder dims.
        base = config.hidden_size
        self.latent_dim = _resolve_dim(latent_dim, base)
        self.ae_hidden = _resolve_dim(ae_hidden, base)
        self.kl_beta = kl_beta
        self.kl_clip = kl_clip
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.ae_dropout = ae_dropout

        self.noise_dim = _resolve_dim(noise_dim, base)
        self.energy_blocks = energy_blocks
        self.energy_samples_n = energy_samples_n
        self.energy_samples_m = energy_samples_m
        self.energy_alpha = energy_alpha
        self.energy_warmup_steps = int(energy_warmup_steps)
        self.vote_num_samples = int(vote_num_samples)

        # Persistent step counter. Buffer so it survives checkpoint/resume:
        # restarting the warmup after a resume would re-destabilize the model.
        self.register_buffer(
            "_train_step", torch.zeros((), dtype=torch.long), persistent=True
        )

        # Transient diagnostic stash for training_metrics(). Floats updated
        # during encode() / _register_energy_loss; consumed by the dynamics
        # callback. NaN signals "not yet observed this run."
        self._diag: Dict[str, float] = {}

        self.pad_token_id = int(getattr(config, "pad_token_id", 0))

        # CALM owns its embeddings, so it sizes to the tokenizer's true
        # vocabulary (byte_vocab_size, e.g. 264 for byte-level) rather than
        # vocab_size, which byte-latent overloads as the hash-bucket count.
        self._output_vocab_size = (
            getattr(config, "byte_vocab_size", None) or config.vocab_size
        )

        self.vae = CALMVAE(
            vocab_size=self._output_vocab_size,
            embed_dim=config.embed_size,
            chunk_size=self.K,
            latent_dim=self.latent_dim,
            hidden_dim=self.ae_hidden,
            dropout=self.ae_dropout,
        )

        # The token classifier (forward/crystal/...) is built from
        # HEAD_REGISTRY and injected via set_head(); CALM applies it to the
        # VAE decoder features. Stored as a bare ref (in a list) so the head
        # stays owned by the model and isn't double-registered here.
        self._head: list = []

        # LM input path (independent of the VAE, matching the reference):
        # K token embeddings → embed_proj → one hidden_size vector per patch.
        # Kept separate from the VAE's tok_emb so the AR/energy loss can't
        # leak gradient into the codec via a shared embedding.
        self.lm_tok_emb = nn.Embedding(self._output_vocab_size, config.embed_size)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.K * config.embed_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6),
        )

        self.energy_head = EnergyHead(
            cond_dim=config.hidden_size,
            noise_dim=self.noise_dim,
            latent_dim=self.latent_dim,
            hidden_dim=max(config.hidden_size, self.latent_dim),
            num_blocks=self.energy_blocks,
        )

        # Loss side-channel: PraxisModel consumes these after decode().
        self._pending_losses: Dict[str, torch.Tensor] = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"K={self.K}, "
            f"latent={self.latent_dim}, "
            f"ae_hidden={self.ae_hidden}, "
            f"energy_blocks={self.energy_blocks}, "
            f"N={self.energy_samples_n}, "
            f"M={self.energy_samples_m})"
        )

    # ------------------------------------------------------------------
    # Encoder-interface surface
    # ------------------------------------------------------------------

    # Output layout the injected LM head sizes its classifier to: the VAE
    # decoder emits features at ae_hidden over the true token vocabulary.
    @property
    def output_dim(self) -> int:
        return self.ae_hidden

    @property
    def output_vocab_size(self) -> int:
        return self._output_vocab_size

    def set_head(self, head: nn.Module) -> None:
        """Receive the LM head built from HEAD_REGISTRY. Held as a bare ref
        (the model owns the parameters); CALM applies it to decoder features."""
        self._head = [head]

    def _classify(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project VAE decoder features to token logits via the injected head."""
        if not self._head:
            raise RuntimeError(
                "CALMEncoder has no LM head; the model must call set_head()."
            )
        return self._head[0](hidden)

    @property
    def classifier(self) -> Optional[nn.Module]:
        """Projection module of the injected head (used by cut-CE paths)."""
        return self._head[0].classifier if self._head else None

    @property
    def outputs_are_aligned(self) -> bool:
        """The reconstructed logits are returned aligned token-for-token,
        but the main CE path is bypassed via ``handles_loss``."""
        return True

    @property
    def handles_loss(self) -> bool:
        """Skip the default CE path; we register losses internally."""
        return True

    # sequence_length_multiplier defaults to 1 (BaseEncoder): the global
    # transformer sees patches, not tokens.

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _pad_to_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        rem = L % self.K
        if rem == 0:
            return input_ids
        pad_len = self.K - rem
        pad = input_ids.new_full((B, pad_len), self.pad_token_id)
        return torch.cat([input_ids, pad], dim=1)

    def encode(
        self,
        input_ids: torch.Tensor,
        block_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """See ``BaseEncoder``.

        Returns ``(patch_embeds, h_encoder, patch_lengths, block_ids,
        encoder_loss, local_decoder_tokens)``. ``h_encoder`` is ``None``
        because CALM does not have a byte-level local encoder. The
        ``block_ids`` argument is accepted for API parity but unused -
        CALM compresses K tokens into one latent and has no byte-level
        path that would benefit from per-token isolation.
        """
        padded = self._pad_to_chunk(input_ids)
        B, L = padded.shape
        N = L // self.K

        mean, logvar = self.vae.encode(padded)  # [B, N, latent_dim]
        z = self.vae.reparameterize(mean, logvar)

        recon_hidden = self.vae.decode(z)  # [B, N*K, ae_hidden]
        recon_logits = self._classify(recon_hidden)  # [B, N*K, V]
        recon_loss = F.cross_entropy(
            recon_logits.reshape(-1, self.vae.vocab_size),
            padded.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        kl = self.vae.kl_divergence(mean, logvar, per_dim_clip=self.kl_clip).mean()
        beta_t = self._current_kl_beta()
        # K-scaled recon to match the reference's stage-1 AE training: their
        # objective is `loss * patch_size + kl_loss * kl_weight`, so β is K
        # times softer relative to recon than a plain mean. Without this our
        # effective β at K=4 is ~4x harder, biasing toward posterior collapse.
        encoder_loss = self.K * recon_loss + beta_t * kl

        # Stash what decode() needs.
        self._last_padded = padded
        self._last_mean = mean
        self._last_logvar = logvar
        self._last_z = z
        self._last_N = N
        # Per-byte recon CE for val_bits_per_byte (training-objective loss is
        # K-scaled per-patch and would give a K-times-too-large bpb).
        self._last_recon_loss = recon_loss.detach()

        self._stash_diagnostics(mean, logvar, recon_loss, kl, beta_t)
        if self.training:
            self._train_step += 1

        # LM input = embed_proj(K token embeddings per patch). Independent
        # of z, so the AR/energy loss naturally can't reach the VAE encoder.
        tok_e = self.lm_tok_emb(padded)  # [B, N*K, embed_size]
        tok_e = tok_e.reshape(B, N, self.K * tok_e.size(-1))
        patch_embeds = self.embed_proj(tok_e)  # [B, N, hidden_size]
        patch_lengths = padded.new_full((B, N), self.K, dtype=torch.long)

        # Return None for block_ids: patch-level boundaries do not map
        # cleanly back to the original token-level sequence boundaries
        # once K-token compression has been applied.
        return patch_embeds, None, patch_lengths, None, encoder_loss, padded

    def decode(
        self,
        h: torch.Tensor,
        h_encoder: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        local_decoder_tokens: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstructed token logits and register the energy loss.

        ``h`` is the global transformer's hidden state per-patch
        (``[B, N, hidden_size]``). The standard reconstruction path uses
        the posterior sample stashed in ``encode``; the energy path
        derives next-latent proposals from ``h[:, :-1]`` and compares
        them against posterior samples of patch ``[:, 1:]``.
        """
        z = self._last_z  # posterior sample used for recon logits
        recon_hidden = self.vae.decode(z)
        recon_logits = self._classify(recon_hidden)

        # Unconditional call: gating happens inside the dynamo-disabled body.
        # Gating *here* puts the graph-break under a Python branch and
        # produces SpeculationLogDivergence on dynamo retries.
        self._register_energy_loss(h)

        return recon_logits, recon_hidden

    @dynamo.disable()
    def _register_energy_loss(self, h: torch.Tensor) -> None:
        """Energy-score loss between next-position model samples and
        next-position posterior samples. See module docstring.

        Runs eager: the pairwise-distance reshapes over a dynamic patch
        count produce symbolic kernels Inductor can't codegen (CantSplit
        on ``M*L*(N-1)`` vs ``N_samples*(N-1)``). The global transformer
        still compiles around this graph break. Gating lives *inside*
        this function (rather than at the call site) so the graph break
        isn't under a Python branch in the compiled caller, which
        produces SpeculationLogDivergence on dynamo retries.
        """
        if not self.training:
            return
        B, N = h.shape[0], h.shape[1]
        if N < 2:
            return
        # VAE warmup: skip energy loss until the codec has stabilized, so
        # the energy head isn't chasing a wildly-moving target.
        if int(self._train_step.item()) < self.energy_warmup_steps:
            return

        # Position p predicts latent at p+1. h_cond: [B, N-1, hidden]
        h_cond = h[:, :-1, :]

        # Target: posterior samples for patches 1..N-1. M independent
        # draws per position, stop-grad so the VAE does not see the
        # energy path.
        mean_t = self._last_mean[:, 1:, :].detach()
        logvar_t = self._last_logvar[:, 1:, :].detach()
        M = self.energy_samples_m
        std_t = (0.5 * logvar_t).exp()
        eps_t = torch.randn(M, *mean_t.shape, device=mean_t.device, dtype=mean_t.dtype)
        # [M, B, N-1, L] -> [B, N-1, M, L]
        target_samples = (mean_t.unsqueeze(0) + std_t.unsqueeze(0) * eps_t).permute(
            1, 2, 0, 3
        )

        # Model samples: N draws from energy head. Same reshape.
        N_samples = self.energy_samples_n
        # [N, B, N-1, L] -> [B, N-1, N, L]
        model_raw = self.energy_head.sample(h_cond, num_samples=N_samples).permute(
            1, 2, 0, 3
        )

        loss = energy_score_loss(model_raw, target_samples)
        scaled = self.energy_alpha * loss
        self._pending_losses["energy"] = scaled
        self._diag["calm_energy_loss"] = float(scaled.detach())

    # ------------------------------------------------------------------
    # Loss side-channel
    # ------------------------------------------------------------------

    def consume_pending_losses(self) -> Dict[str, torch.Tensor]:
        """Pop any losses registered during the last ``decode`` call."""
        out = self._pending_losses
        self._pending_losses = {}
        return out

    def per_byte_val_loss(self) -> Optional[torch.Tensor]:
        """Most recent per-byte reconstruction CE for val_bits_per_byte.

        The training-objective ``encoder_loss`` is K-scaled per-patch; using
        it in ``_compute_bits_per_byte`` would over-report by a factor of K
        and wouldn't be comparable to byte-latent's bpb. Codec recon CE is
        the natural per-byte signal.
        """
        return getattr(self, "_last_recon_loss", None)

    # ------------------------------------------------------------------
    # Warmup + diagnostics
    # ------------------------------------------------------------------

    def _current_kl_beta(self) -> float:
        """Linearly annealed β: 0 → kl_beta over kl_warmup_steps, then held."""
        if self.kl_warmup_steps <= 0:
            return float(self.kl_beta)
        progress = float(self._train_step.item()) / float(self.kl_warmup_steps)
        return float(self.kl_beta) * min(1.0, max(0.0, progress))

    @torch.no_grad()
    def _stash_diagnostics(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        recon_loss: torch.Tensor,
        kl_mean: torch.Tensor,
        beta_t: float,
    ) -> None:
        """Update transient values exposed via training_metrics().

        Cheap reductions over the current batch; consumed by the dynamics
        callback. Values are floats so they're trivially JSON-serializable
        through the dynamics logger.
        """
        # Pre-clip per-dim KL: identifies dims still being squeezed to the
        # free-bits floor (dead dims).
        per_dim_raw = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
        active_frac = (per_dim_raw > self.kl_clip).float().mean()
        std = (0.5 * logvar).exp()
        denom = beta_t * float(kl_mean.detach()) + 1e-9
        ratio = float(recon_loss.detach()) / denom

        self._diag["calm_latent_norm_mean"] = float(mean.norm(dim=-1).mean())
        self._diag["calm_latent_std_mean"] = float(std.mean())
        self._diag["calm_kl_active_frac"] = float(active_frac)
        self._diag["calm_recon_kl_ratio"] = ratio
        self._diag["calm_kl_beta"] = float(beta_t)
        # Energy loss is updated in _register_energy_loss; default 0.0 during
        # warmup or eval (when it's not registered).
        self._diag.setdefault("calm_energy_loss", 0.0)

    def training_metrics(self) -> Dict[str, float]:
        """Diagnostic scalars for the Dynamics tab. See ``metric_descriptions``.

        Co-located with the math that produces them (per the project
        convention); the DynamicsLogger callback picks them up.
        """
        return dict(self._diag)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _patch_vote_sample(
        self,
        h_last: torch.Tensor,
        temperature: float,
        num_samples: int = 200,
    ) -> torch.Tensor:
        """Paper Algorithm 1 (temperature sampling for CALM).

        ``temperature`` must be in (0, 1]; for T<1 we draw ``num_samples``
        candidate latents, decode each to an argmax K-token patch, count
        exact patch matches, and pick the most-supported patch via
        combinatorial weighting (cascade from n_initial = round(1/T) down).
        For T=1 a single latent is sampled - matching the reference.

        Args:
            h_last: ``[hidden_size]`` conditioning (single stream).
            temperature: T in (0, 1]. Strict T = 1/integer in the paper;
                we round 1/T to the nearest integer so any T in range works.
            num_samples: pool size for voting.

        Returns:
            ``[K]`` selected token ids.
        """
        if temperature >= 1.0:
            z = self.energy_head.sample(h_last, num_samples=1)  # [1, latent_dim]
            z = z.view(1, 1, -1)
            recon_hidden = self.vae.decode(z)
            recon_logits = self._classify(recon_hidden)
            return recon_logits.argmax(dim=-1).view(self.K)

        n_initial = max(1, int(round(1.0 / max(temperature, 1e-6))))

        # Pool of candidate patches: decode each candidate latent and take
        # argmax tokens. Stochasticity lives in the latent draws; voting
        # then concentrates on the patches the head agrees on most often.
        z_pool = self.energy_head.sample(h_last, num_samples=num_samples)
        z_pool = z_pool.view(num_samples, 1, -1)
        recon_hidden = self.vae.decode(z_pool)  # [N, K, ae_hidden]
        recon_logits = self._classify(recon_hidden)  # [N, K, V]
        patches = recon_logits.argmax(dim=-1)  # [N, K]

        counts = Counter(tuple(p.tolist()) for p in patches)

        # Cascade: try n_initial, n_initial - 1, ..., 1. Pick from patches
        # appearing >= n times, weighted by C(count, n).
        for n in range(n_initial, 0, -1):
            candidates = {p: c for p, c in counts.items() if c >= n}
            if candidates:
                weights = [math.comb(c, n) for c in candidates.values()]
                chosen = random.choices(list(candidates.keys()), weights=weights, k=1)[0]
                return torch.tensor(chosen, dtype=torch.long, device=h_last.device)

        # Defensive fallback (n=1 must find a match unless num_samples=0).
        return patches[0]

    @torch.no_grad()
    def custom_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        base_forward,
        generation_config=None,
        **kwargs,
    ):
        """CALM generation: latent LM -> energy head -> VAE decode.

        Each step runs the global transformer over the latent prefix
        (``base_forward``), then uses patch-vote temperature sampling
        (paper Algorithm 1) to pick the next K-token patch: draw a pool
        of candidate latents, decode each to an argmax patch, and select
        by combinatorial voting on exact patch matches. Stops on EOS or
        ``max_new_tokens``.
        """
        from types import SimpleNamespace

        if inputs is None:
            raise ValueError("CALM generate requires an input_ids prompt")

        max_new_tokens = getattr(generation_config, "max_new_tokens", 100) or 100
        temperature = getattr(generation_config, "temperature", 1.0) or 1.0
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        # Pool size for patch-vote (T<1). Profile default lives in
        # self.vote_num_samples (paper-scale = 200); generation_config can
        # still override per-call. Smaller pools = faster but noisier votes.
        num_samples = int(
            getattr(generation_config, "calm_num_samples", None)
            or self.vote_num_samples
        )
        return_dict = kwargs.get("return_dict_in_generate", False)

        eos_set = set()
        if isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        elif isinstance(eos_token_id, (list, tuple)):
            eos_set = set(eos_token_id)

        generated = inputs
        num_new = 0
        done = False

        while num_new < max_new_tokens and not done:
            base_out = base_forward(generated)
            h_last = base_out.last_hidden_state[:, -1, :]  # [B, hidden]

            new_tokens = self._patch_vote_sample(
                h_last[0], temperature=float(temperature), num_samples=num_samples
            )  # [K]
            new_tokens = new_tokens.view(1, self.K)

            # Expand batch dim if the prompt had batch > 1 (rare for CLI).
            if generated.size(0) > 1:
                new_tokens = new_tokens.expand(generated.size(0), -1)

            generated = torch.cat([generated, new_tokens], dim=1)
            num_new += self.K

            if eos_set:
                for t in new_tokens.view(-1).tolist():
                    if t in eos_set:
                        done = True
                        break

        if return_dict:
            return SimpleNamespace(sequences=generated)
        return generated
