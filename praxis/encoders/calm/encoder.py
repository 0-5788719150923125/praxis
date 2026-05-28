"""CALM encoder: VAE + energy head, exposed via the Praxis encoder interface.

See ``README.md`` for the full architecture rationale. Briefly:

1. ``encode`` compresses K-token chunks to continuous latents. The
   posterior sample ``z`` is projected to ``hidden_size`` and fed to the
   global transformer as the "patch embedding" sequence. The VAE's own
   reconstruction + KL objective is returned as ``encoder_loss`` and
   composes with the rest of the loss container.
2. The global transformer autoregresses over latents.
3. ``decode`` uses the LM hidden state at position ``p`` to drive an
   energy head that produces proposals for latent ``p+1``; those are
   compared against the posterior samples of ``p+1`` under the
   energy-score loss. The reconstructed-token logits are returned for
   sanity checking but do not participate in the main loss (the encoder
   sets ``handles_loss=True`` to bypass the default CE path).
"""

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

        # Latent → hidden_size projection for the global transformer.
        self.latent_in = nn.Linear(self.latent_dim, config.hidden_size, bias=False)

        self.energy_head = EnergyHead(
            cond_dim=config.hidden_size,
            noise_dim=self.noise_dim,
            latent_dim=self.latent_dim,
            hidden_dim=max(config.hidden_size, self.latent_dim),
            num_blocks=self.energy_blocks,
            dropout=self.ae_dropout,
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
        encoder_loss = recon_loss + beta_t * kl

        # Stash what decode() needs.
        self._last_padded = padded
        self._last_mean = mean
        self._last_logvar = logvar
        self._last_z = z
        self._last_N = N

        self._stash_diagnostics(mean, logvar, recon_loss, kl, beta_t)
        if self.training:
            self._train_step += 1

        # Global transformer inputs: one token per patch.
        patch_embeds = self.latent_in(z)  # [B, N, hidden_size]
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

        if self.training and h.size(1) >= 2:
            self._register_energy_loss(h)

        return recon_logits, recon_hidden

    @dynamo.disable()
    def _register_energy_loss(self, h: torch.Tensor) -> None:
        """Energy-score loss between next-position model samples and
        next-position posterior samples. See module docstring.

        Runs eager: the pairwise-distance reshapes over a dynamic patch
        count produce symbolic kernels Inductor can't codegen (CantSplit
        on ``M*L*(N-1)`` vs ``N_samples*(N-1)``). The global transformer
        still compiles around this graph break.
        """
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
        (``base_forward``), samples a latent from the energy head conditioned
        on the last hidden state (LF-temperature if T != 1), decodes it
        through the VAE to K tokens, and appends. Stops on EOS or
        ``max_new_tokens``.
        """
        from types import SimpleNamespace

        from praxis.generation.lf_temperature import lf_temperature_sample_batched

        if inputs is None:
            raise ValueError("CALM generate requires an input_ids prompt")

        max_new_tokens = getattr(generation_config, "max_new_tokens", 100) or 100
        temperature = getattr(generation_config, "temperature", 1.0) or 1.0
        eos_token_id = getattr(generation_config, "eos_token_id", None)
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

            def sampler(n: int, _h=h_last) -> torch.Tensor:
                # Draw n latents from the energy head for batch-index 0;
                # generation is single-stream for now.
                return self.energy_head.sample(_h[0], num_samples=n)

            z_new = lf_temperature_sample_batched(
                sampler, temperature=float(temperature), num_candidates=64
            )  # [latent_dim]
            z_new = z_new.view(1, 1, -1)

            recon_hidden = self.vae.decode(z_new)  # [1, K, ae_hidden]
            recon_logits = self._classify(recon_hidden)  # [1, K, V]
            # Greedy token choice per position inside the patch; the
            # stochasticity lives in the latent draw, not the tokens.
            new_tokens = recon_logits.argmax(dim=-1).view(1, self.K)

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
