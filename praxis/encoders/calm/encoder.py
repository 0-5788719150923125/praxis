"""CALM encoder: VAE + energy head, exposed via the Praxis encoder interface.

Architecture (matches the reference at github.com/shaochenze/calm):

1. ``encode`` builds the LM's per-patch input from raw token embeddings,
   *not* from the VAE: ``embed_proj`` compresses K token embeddings into
   one ``hidden_size`` vector per patch. The VAE encoder runs in
   parallel to produce the energy-score's posterior targets, and the VAE
   decoder + classifier produce the AE's own reconstruction loss
   (composed into ``encoder_loss``). The LM input is decoupled from the
   VAE's latent quality, mirroring the reference (where the AE is
   frozen during LM training and only emits targets). With
   ``ae_freeze_steps`` set, this freezing is real and two-staged: stage
   1 trains only the codec, then the codec freezes and stage 2 trains
   only the LM/energy head against a now-stationary latent target.
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
from praxis.losses import get_loss_function
from praxis.losses.energy_score import energy_score_loss

from ..base import BaseEncoder
from .vae import CALMVAE

# AE pretraining-phase convergence detector. Fixed and model-agnostic (per the
# no-per-experiment-tuning rule): the phase ends when reconstruction stops
# improving - the recon curve's linear trend across the window shrinks below the
# window's own noise, sustained for PRETRAIN_PATIENCE readings and only after the
# LR warmup horizon - or the max-steps cap is hit as a backstop. The warmup floor
# matters: during the LR ramp a flat recon curve is just the low LR, not convergence.
#
# Trend-vs-noise, NOT relative-to-mean. The gate is |slope*n| / std(recon) over
# the window: the total linear drift in units of the per-sample noise. A
# relative-to-mean delta (the old form) exploded as recon CE -> 0 - the noise
# floor stays put while the mean vanishes, so it could never read "converged" on
# a good codec. Trend-vs-noise is bounded (drift and noise shrink together) and
# self-normalizes against outliers (a spike inflates the denominator too).
PRETRAIN_WINDOW = 256
PRETRAIN_FLAT_EPS = 1.0  # converged when the window's drift < one noise std

# Conditioning-anchor weight for the energy head. The energy score alone is a
# weak, high-variance signal: at small scale the head learns the right marginal
# latent scale but a condition-weak, high-variance conditional (verified: given
# real context, cross-sample patch agreement ~0, correct-next-patch rate ~0), so
# generation samples the marginal (low T repeats the mode, high T is gibberish).
# A direct MSE from the conditioning onto the next posterior mean gives the
# strong gradient the score lacks. Weighted comparable-to / above the energy
# term (~O(5)) so it dominates the early steering - the score alone never
# concentrated the head over thousands of steps - then fades as the MSE -> 0.
# Deviates from the paper (justified by the scale gap the paper never faces);
# set 0.0 for paper-pure behavior. Watch calm_energy_anchor descend.
ENERGY_ANCHOR_WEIGHT = 5.0
# Start emitting / computing the convergence Δ once this many recon samples
# exist, so the chart tracks the descent from early on instead of staying blank
# (then flat) until the full window fills. The freeze decision still waits for
# the full window - an early, small-sample Δ is informative but too noisy to
# latch on.
PRETRAIN_MIN_SAMPLES = 16
# Consecutive below-threshold readings required before convergence latches. The
# window slides one sample per microbatch, so this demands the plateau hold for
# a sustained stretch (a full extra window) rather than ending the phase on a
# single lucky read at the moment the window first fills.
PRETRAIN_PATIENCE = PRETRAIN_WINDOW


def _resolve_dim(spec: Union[int, float], base: int) -> int:
    """``float`` -> fraction of ``base`` (e.g. 0.5); ``int`` -> absolute size."""
    if isinstance(spec, float):
        return max(1, round(spec * base))
    return int(spec)


class CALMEncoder(BaseEncoder):
    # Builds its own lm_tok_emb sized to the tokenizer's true (byte) vocab, so
    # vocab_size should report that width, not the hash-bucket count.
    owns_embeddings = True

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
                "samples. Zero during the codec-only stage; should trend down "
                "once active."
            ),
            "chart": {
                "title": "Energy Loss",
                "y_label": "energy",
                "y_scale": "linear",
                "group": "calm",
                "order": 60,
            },
        },
        "calm_energy_anchor": {
            "description": (
                "MSE of the head's zero-noise prediction vs the next posterior "
                "mean - the conditioning anchor that forces the conditional to "
                "concentrate on the correct next latent. Should descend; if it "
                "stays high the head isn't learning next-patch direction."
            ),
            "chart": {
                "title": "Energy Conditioning Anchor (MSE)",
                "y_label": "mse",
                "y_scale": "logarithmic",
                "group": "calm",
                "order": 64,
            },
        },
        "calm_energy_cond_gap": {
            "description": (
                "Energy of context-mismatched targets minus matched targets. "
                ">0 means the head USES the conditioning (next-patch prediction "
                "beats marginal); ~0 means it's ignoring context and not learning "
                "sequences. The decisive signal for whether CALM is modeling order."
            ),
            "chart": {
                "title": "Energy Conditioning Gap",
                "y_label": "mismatch - matched",
                "y_scale": "linear",
                "group": "calm",
                "order": 65,
            },
        },
        "calm_ae_frozen": {
            "description": (
                "1 once the codec is frozen and stage 2 (energy head only) "
                "begins; 0 during the codec-training stage. Flat 0 means "
                "two-stage training is off (legacy joint mode)."
            ),
            "chart": {
                "title": "Codec Frozen (Stage 2)",
                "y_label": "frozen",
                "y_scale": "linear",
                "group": "calm",
                "order": 70,
            },
        },
        "calm_recon_ce": {
            "description": (
                "Raw per-step codec reconstruction CE during AE pretraining. "
                "Its plateau is what the convergence detector watches; once the "
                "codec freezes this stops updating."
            ),
            "chart": {
                "title": "Codec Recon CE (pretrain)",
                "y_label": "recon CE",
                "y_scale": "logarithmic",
                "group": "calm",
                "order": 80,
            },
        },
        "calm_pretrain_flatness": {
            "description": (
                "Codec convergence detector: the recon curve's linear trend "
                "across the window in units of the window's own noise "
                "(|slope*n| / std). Scale-free and bounded as recon CE -> 0. "
                "Tracked from the first few samples; the latch waits for a full "
                "window. The codec freezes once this holds below the flat "
                "threshold (1.0 = trend smaller than one noise std - the trend "
                "is lost in the noise) for a full patience window."
            ),
            "chart": {
                "title": "Pretrain Convergence (trend/noise)",
                "y_label": "|trend| / noise",
                "y_scale": "logarithmic",
                "group": "calm",
                "order": 90,
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
        kl_warmup_steps: Optional[int] = None,
        ae_dropout: float = 0.15,
        noise_dim: Union[int, float] = 128,
        energy_blocks: int = 3,
        energy_samples_n: int = 8,
        energy_samples_m: int = 100,
        energy_alpha: float = 1.0,
        energy_warmup_steps: int = 0,
        ae_freeze_steps: Optional[int] = None,
        requires_pretraining: bool = True,
        ae_max_pretrain_steps: int = 20000,
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
        self.ae_dropout = ae_dropout

        # All schedules below are in optimizer-step units. _train_step counts
        # encode() calls (microbatches), so the gates divide by grad_accumulation
        # to convert - see _opt_step(). When kl_warmup/ae_freeze are left None,
        # they lock to the LR warmup horizon: the codec trains entirely under the
        # rising-LR phase and freezes at the LR peak, then the rest of the model
        # adapts to a fixed codec during decay.
        self._grad_accum = max(1, int(getattr(config, "grad_accumulation", 1) or 1))
        warmup = int(getattr(config, "warmup_steps", 0) or 0)
        self.kl_warmup_steps = (
            warmup if kl_warmup_steps is None else int(kl_warmup_steps)
        )
        # Convergence can't latch until the LR warmup horizon has elapsed: while
        # the LR is still ramping from ~0, a flat recon curve is just the low LR,
        # not real convergence. Without this floor the codec "converges" in a few
        # dozen steps the moment the window first fills. Optimizer-step units.
        self._pretrain_min_steps = warmup

        self.noise_dim = _resolve_dim(noise_dim, base)
        self.energy_blocks = energy_blocks
        self.energy_samples_n = energy_samples_n
        self.energy_samples_m = energy_samples_m
        self.energy_alpha = energy_alpha
        self.energy_warmup_steps = int(energy_warmup_steps)
        # Two-stage boundary (optimizer steps): trains the codec alone until
        # this step, then freezes it and trains only the LM/energy head. None
        # locks it to the warmup boundary (co-terminating with the KL anneal);
        # 0 = legacy joint training.
        # When True (default), the codec trains until its reconstruction
        # converges, then freezes; the energy LM trains after. Convergence is
        # the boundary - ae_freeze_steps stays 0 (no fixed cut) unless given
        # explicitly. ae_max_pretrain_steps caps the phase as a backstop.
        # False = legacy joint mode, where the freeze cuts at ae_freeze_steps
        # (defaulting to the LR warmup horizon).
        self.requires_pretraining = bool(requires_pretraining)
        if ae_freeze_steps is not None:
            # Explicit boundary opts into legacy behavior (joint mode when 0,
            # staged-at-step when >0); the convergence detector is disabled.
            self.ae_freeze_steps = int(ae_freeze_steps)
            self.requires_pretraining = False
        elif self.requires_pretraining:
            self.ae_freeze_steps = 0  # convergence-driven freeze
        else:
            self.ae_freeze_steps = warmup
        self.ae_max_pretrain_steps = int(ae_max_pretrain_steps)
        self._recon_hist: list = []
        # Consecutive below-threshold readings so far (see PRETRAIN_PATIENCE).
        # Plain attribute, not a buffer: resetting to 0 on resume is safe - it
        # just re-requires the plateau to re-establish, which is conservative.
        self._pretrain_patience = 0
        # Set once when the codec is frozen, to log the transition a single
        # time. Not a buffer: re-deriving frozen-ness from _train_step on
        # resume is what re-applies the freeze, so this flag may reset freely.
        self._ae_frozen_logged = False
        self.vote_num_samples = int(vote_num_samples)

        # Persistent step counter. Buffer so it survives checkpoint/resume:
        # restarting the warmup after a resume would re-destabilize the model.
        self.register_buffer(
            "_train_step", torch.zeros((), dtype=torch.long), persistent=True
        )

        # Latched once AE pretraining converges (persistent so resume does not
        # restart the phase). Drives the codec freeze and the energy-loss gate.
        self.register_buffer(
            "_pretrain_done", torch.zeros((), dtype=torch.bool), persistent=True
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

        # Codec reconstruction obeys config.loss_func like every other path -
        # cross_entropy by default, but a distance/centroid loss (HALO) trains
        # the VAE decoder against the head's centroids instead. Built over the
        # codec's true output vocab. See _reconstruction_loss.
        self.recon_loss_fn = get_loss_function(
            config.loss_func, self._output_vocab_size
        )

        # "harmonic" swaps the codec's scalar dropout for a standing-wave field
        # over (patch position, channel) - see HarmonicDropout. Stage-1 only;
        # the field vanishes once the codec freezes (eval disables it).
        self.vae = CALMVAE(
            vocab_size=self._output_vocab_size,
            embed_dim=config.embed_size,
            chunk_size=self.K,
            latent_dim=self.latent_dim,
            hidden_dim=self.ae_hidden,
            dropout=self.ae_dropout,
            dropout_mode=getattr(config, "ae_dropout_mode", "scalar"),
            dropout_cycles=getattr(config, "ae_dropout_cycles", 2),
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
    # Autoencoder pretraining phase (BaseEncoder contract)
    # ------------------------------------------------------------------

    def in_pretraining(self) -> bool:
        """True until the codec's reconstruction converges (or the cap is hit).
        While True the model locks everything but the VAE and trains only
        ``pretraining_loss`` - the global transformer never runs."""
        return self.requires_pretraining and not bool(self._pretrain_done.item())

    def _codec_parameters(self):
        """The full autoencoder: VAE + the head's reconstruction-path params
        (token classifier, and any feature field like crystal_harmonic).

        These train during AE pretraining and ONLY then - in phase 2 the recon
        CE is zeroed, so a classifier left at random init here would stay there
        and emit garbage tokens forever. The head's own params are included, but
        the LM-side encoder modules (energy head, embed_proj, lm_tok_emb) are
        excluded by id so a head->encoder back-ref can't drag them in (see
        reference_head_encoder_backref); iterating head.parameters() can recurse
        into this encoder, so we filter rather than trust the boundary.
        """
        exclude = set()
        for m in (self.energy_head, self.embed_proj, self.lm_tok_emb):
            for p in m.parameters():
                exclude.add(id(p))
        seen = set()
        for p in self.vae.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                yield p
        if self._head:
            for p in self._head[0].parameters():
                if id(p) in exclude or id(p) in seen:
                    continue
                seen.add(id(p))
                yield p

    def training_stage(self) -> str:
        """ "preflight" while the codec pretrains in isolation, then "pretrain"
        once the codec is frozen and the energy LM trains."""
        return "preflight" if self.in_pretraining() else "pretrain"

    def pretraining_parameters(self):
        """Codec params (VAE + head recon path) train during the AE warmup."""
        return self._codec_parameters()

    def freeze_after_pretraining(self) -> None:
        """One-shot codec freeze at the phase transition. ``_set_codec_mode``
        re-applies the VAE freeze every step in phase 2 (surviving Lightning's
        per-step train() and checkpoint resume). We eval() only the VAE (the
        recon-path dropout lives there); calling head.eval() would recurse into
        this encoder via the back-ref and wrongly eval the global transformer."""
        self.vae.eval()
        for p in self._codec_parameters():
            p.requires_grad_(False)

    def _reconstruction_loss(
        self,
        recon_logits: torch.Tensor,
        padded: torch.Tensor,
        recon_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Codec reconstruction loss via the configured loss function.

        Flattens the per-patch outputs and routes through ``recon_loss_fn``.
        Pad positions map to -100 (the registry losses' ignore index), which
        reproduces the old ``ignore_index=pad_token_id`` behavior for CE.
        Centroid losses (HALO) also get the decoder features as ``embeddings``
        and the head's classifier centroids.
        """
        V = self.vae.vocab_size
        flat_logits = recon_logits.reshape(-1, V)
        flat_labels = padded.reshape(-1).clone()
        flat_labels[flat_labels == self.pad_token_id] = -100
        flat_emb = recon_hidden.reshape(-1, recon_hidden.shape[-1])
        classifier = self._head[0].classifier if self._head else None
        return self.recon_loss_fn(
            logits=flat_logits,
            labels=flat_labels,
            embeddings=flat_emb,
            classifier=classifier,
            input_ids=flat_labels,
        )

    @torch.no_grad()
    def _recon_ce(
        self, recon_logits: torch.Tensor, padded: torch.Tensor
    ) -> torch.Tensor:
        """Detached reconstruction CE for the bits-per-byte fidelity metric,
        kept independent of the training loss so ``val_codec_bpb`` stays in
        bits even when reconstruction trains under a non-CE objective."""
        return F.cross_entropy(
            recon_logits.reshape(-1, self.vae.vocab_size),
            padded.reshape(-1),
            ignore_index=self.pad_token_id,
        )

    def pretraining_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Codec-only objective: K-scaled reconstruction CE + annealed
        free-bits KL. The global transformer and energy head are not run."""
        padded = self._pad_to_chunk(input_ids)
        mean, logvar = self.vae.encode(padded)
        z = self.vae.reparameterize(mean, logvar)
        recon_hidden = self.vae.decode(z)
        recon_logits = self._classify(recon_hidden)
        recon_loss = self._reconstruction_loss(recon_logits, padded, recon_hidden)
        kl = self.vae.kl_divergence(mean, logvar, per_dim_clip=self.kl_clip).mean()
        beta_t = self._current_kl_beta()
        loss = self.K * recon_loss + beta_t * kl

        self._train_step += 1
        self._last_recon_loss = self._recon_ce(recon_logits, padded)
        self._stash_diagnostics(mean, logvar, recon_loss, kl, beta_t)
        self._diag["calm_ae_frozen"] = 0.0
        self._update_pretrain_convergence(float(recon_loss.detach()))
        return loss

    @dynamo.disable()
    def _update_pretrain_convergence(self, recon: float) -> None:
        """Latch ``_pretrain_done`` when recon stops improving or the cap is
        reached. Convergence = relative drop across the window below EPS."""
        if bool(self._pretrain_done.item()):
            return
        self._recon_hist.append(recon)
        if len(self._recon_hist) > PRETRAIN_WINDOW:
            self._recon_hist.pop(0)
        self._diag["calm_recon_ce"] = recon
        step = self._opt_step()
        if step >= self.ae_max_pretrain_steps:
            self._mark_pretrain_done(step, "max-steps cap")
            return
        if len(self._recon_hist) >= PRETRAIN_MIN_SAMPLES:
            n = len(self._recon_hist)
            mean_x = (n - 1) / 2.0
            mean_y = sum(self._recon_hist) / n
            cov = sum(
                (i - mean_x) * (v - mean_y) for i, v in enumerate(self._recon_hist)
            )
            var = sum((i - mean_x) ** 2 for i in range(n))
            slope = cov / var if var > 0 else 0.0
            # The window's linear drift measured in units of its own noise:
            # |slope*n| / std(recon). Scale-free and, unlike a relative-to-mean
            # delta, bounded as recon CE -> 0 (drift and noise shrink together),
            # so a true plateau reads "trend lost in the noise". Computed from
            # PRETRAIN_MIN_SAMPLES onward so the chart tracks the descent early;
            # the latch waits for the full window so it can't fire on noise.
            std_y = (sum((v - mean_y) ** 2 for v in self._recon_hist) / n) ** 0.5
            flat = abs(slope * n) / (std_y + 1e-9)
            self._diag["calm_pretrain_flatness"] = flat
            # Latch only after the LR warmup floor, with a full window, once the
            # plateau has held for PRETRAIN_PATIENCE consecutive readings.
            converging = (
                len(self._recon_hist) >= PRETRAIN_WINDOW
                and step >= self._pretrain_min_steps
                and flat < PRETRAIN_FLAT_EPS
            )
            if converging:
                self._pretrain_patience += 1
                if self._pretrain_patience >= PRETRAIN_PATIENCE:
                    self._mark_pretrain_done(step, f"converged (flat={flat:.3f})")
            else:
                self._pretrain_patience = 0

    def _mark_pretrain_done(self, step: int, reason: str) -> None:
        self._pretrain_done.fill_(True)
        print(
            f"[CALM] AE pretraining complete at optimizer step {step} "
            f"({reason}); freezing codec, energy LM begins."
        )

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

        # Stage gate: in two-stage mode the codec freezes after stage 1 so
        # the energy head trains against a stationary target.
        frozen = self._set_codec_mode() if self.training else self._ae_is_frozen()

        mean, logvar = self.vae.encode(padded)  # [B, N, latent_dim]
        z = self.vae.reparameterize(mean, logvar)

        recon_hidden = self.vae.decode(z)  # [B, N*K, ae_hidden]
        recon_logits = self._classify(recon_hidden)  # [B, N*K, V]
        recon_loss = self._reconstruction_loss(recon_logits, padded, recon_hidden)
        kl = self.vae.kl_divergence(mean, logvar, per_dim_clip=self.kl_clip).mean()
        beta_t = self._current_kl_beta()
        # K-scaled recon to match the reference's stage-1 AE training: their
        # objective is `loss * patch_size + kl_loss * kl_weight`, so β is K
        # times softer relative to recon than a plain mean. Without this our
        # effective β at K=4 is ~4x harder, biasing toward posterior collapse.
        if frozen:
            # Stage 2: codec is read-only; contribute nothing to the loss so
            # only the LM/energy head trains.
            encoder_loss = recon_logits.new_zeros(())
        else:
            encoder_loss = self.K * recon_loss + beta_t * kl

        # Stash what decode() needs. The energy gate re-derives frozen-ness
        # from the _train_step buffer rather than a stashed Python flag: under
        # torch.compile an attribute set here (compiled) isn't reliably visible
        # to the dynamo-disabled _register_energy_loss() that reads it, so a
        # stale read would drop the only stage-2 loss and break backward.
        self._last_padded = padded
        self._last_mean = mean
        self._last_logvar = logvar
        self._last_z = z
        self._last_N = N
        # Per-byte codec recon CE for val_codec_bpb (codec fidelity only;
        # see codec_recon_loss). Detached: pure diagnostic.
        self._last_recon_loss = self._recon_ce(recon_logits, padded)

        self._stash_diagnostics(mean, logvar, recon_loss, kl, beta_t)
        self._diag["calm_ae_frozen"] = 1.0 if frozen else 0.0
        if self.training:
            self._train_step += 1
            # Drive the pretraining-phase convergence detector from the
            # always-run codec path: freeze once reconstruction stops
            # improving. (The model-side hard lock that also skips the global
            # transformer during this phase is an optional optimization layered
            # on top via the BaseEncoder pretraining hooks.)
            if self.requires_pretraining and not frozen:
                self._update_pretrain_convergence(float(recon_loss.detach()))

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
        # Runs in eval too: CALM has no CE val loss, so the validation energy
        # score is the only learning signal to surface as val_loss (modeling.py
        # combines registered losses for handles_loss encoders during eval). The
        # eval value is detached below so it never builds a graph.
        B, N = h.shape[0], h.shape[1]
        if N < 2:
            return
        # Skip energy loss until the codec is ready, so the energy head isn't
        # chasing a wildly-moving target. Two-stage: active once the codec is
        # frozen (stage 2). Legacy joint mode: after energy_warmup. Both gates
        # read the _train_step buffer directly (not a stashed flag) so the gate
        # survives torch.compile - see encode(). encode() increments the step
        # before decode() runs, so _ae_is_frozen() here is True whenever encode
        # zeroed encoder_loss, guaranteeing stage 2 always has a grad path.
        if self.requires_pretraining or self.ae_freeze_steps > 0:
            energy_active = self._ae_is_frozen()
        else:
            energy_active = self._opt_step() >= self.energy_warmup_steps
        if not energy_active:
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

        # Diagnostic: does the head actually USE the conditioning? Re-score the
        # same model samples against targets rolled one position out of
        # alignment. If this mismatched energy is no higher than the matched
        # one, the head is ignoring context - modeling the marginal, not the
        # sequence (i.e. "not learning sequences"). Detached; no grad effect.
        if N > 2:
            mismatched = energy_score_loss(
                model_raw.detach(), target_samples.detach().roll(1, dims=1)
            )
            self._diag["calm_energy_cond_gap"] = float((mismatched - loss).detach())

        # Conditioning anchor: regress the head's zero-noise (mean) prediction
        # onto the next posterior mean. A strong, low-variance gradient that
        # forces the conditional to concentrate on the correct next latent -
        # the energy score alone leaves it at the (marginal) scale only.
        total = self.energy_alpha * loss
        if ENERGY_ANCHOR_WEIGHT > 0.0:
            mean_next = self._last_mean[:, 1:, :].detach()
            zero_noise = h_cond.new_zeros(
                *h_cond.shape[:-1], self.energy_head.noise_dim
            )
            anchor = torch.nn.functional.mse_loss(
                self.energy_head(h_cond, zero_noise), mean_next
            )
            total = total + ENERGY_ANCHOR_WEIGHT * anchor
            self._diag["calm_energy_anchor"] = float(anchor.detach())

        # Detach in eval: val_loss only needs the scalar, never a backward graph.
        self._pending_losses["energy"] = total if self.training else total.detach()
        self._diag["calm_energy_loss"] = float((self.energy_alpha * loss).detach())

    # ------------------------------------------------------------------
    # Loss side-channel
    # ------------------------------------------------------------------

    def consume_pending_losses(self) -> Dict[str, torch.Tensor]:
        """Pop any losses registered during the last ``decode`` call."""
        out = self._pending_losses
        self._pending_losses = {}
        return out

    def codec_recon_loss(self) -> Optional[torch.Tensor]:
        """Most recent per-byte codec reconstruction CE.

        This is teacher-forced autoencoder fidelity (encode true tokens,
        decode them back), NOT generation quality - it is near-zero for a
        working codec regardless of whether the LM can generate. The trainer
        surfaces it as ``val_codec_bpb``; trust ``val_brierlm`` for the
        generative path. An energy-based head has no closed-form per-byte
        likelihood, so CALM intentionally does not report ``val_bits_per_byte``.
        """
        return getattr(self, "_last_recon_loss", None)

    # ------------------------------------------------------------------
    # Warmup + diagnostics
    # ------------------------------------------------------------------

    def _opt_step(self) -> int:
        """Optimizer-step count. _train_step counts encode() forward calls
        (microbatches); divide by the accumulation factor so the freeze/anneal
        boundaries align with the LR schedule, which is in optimizer steps."""
        return int(self._train_step.item()) // self._grad_accum

    def _ae_is_frozen(self) -> bool:
        """True once the codec is read-only: after pretraining converges (new
        path), or past the step boundary (legacy joint mode)."""
        if self.requires_pretraining:
            if bool(self._pretrain_done.item()):
                return True
            # Optional explicit boundary (manual override / tests). Normally 0.
            return self.ae_freeze_steps > 0 and self._opt_step() >= self.ae_freeze_steps
        return self.ae_freeze_steps > 0 and (self._opt_step() >= self.ae_freeze_steps)

    @dynamo.disable()
    def _set_codec_mode(self) -> bool:
        """Apply the stage gate during training; return True iff frozen.

        Runs eager (param/eval mutation breaks dynamo). Re-applied every
        step so it survives Lightning's per-step ``model.train()`` (which
        would otherwise re-enable codec dropout) and checkpoint resume
        (``requires_grad`` is not part of the state dict).
        """
        # Seam for a soft landing: today this is a step-function freeze at the
        # LR peak. A future blend would replace the hard requires_grad_(False)
        # below with a codec-LR decay across the early decay phase, gated by how
        # far _opt_step() is past ae_freeze_steps. The boolean gate stays here.
        frozen = self._ae_is_frozen()
        if not frozen:
            return False
        # eval() the VAE each step: kills codec dropout so the energy head's
        # targets (the posterior mean/logvar) stop jittering. requires_grad_
        # (False) means no optimizer updates, so the target distribution is
        # stationary. Only the VAE is touched: the injected head holds a
        # back-reference to this encoder as a submodule, so iterating its
        # parameters() or calling its eval() would recurse back into the
        # encoder and freeze the energy head too. The head needs no explicit
        # freeze anyway - its sole gradient source (the recon CE) is zeroed
        # out in stage 2, so it already stops learning.
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        if not self._ae_frozen_logged:
            print(
                f"[CALM] codec frozen at optimizer step {self._opt_step()}; "
                f"stage 2 (energy head only) begins."
            )
            self._ae_frozen_logged = True
        return True

    def _current_kl_beta(self) -> float:
        """Linearly annealed β: 0 → kl_beta over kl_warmup_steps, then held."""
        if self.kl_warmup_steps <= 0:
            return float(self.kl_beta)
        progress = float(self._opt_step()) / float(self.kl_warmup_steps)
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
        out = dict(self._diag)
        # A non-CE reconstruction loss (HALO) carries its own diagnostics;
        # fold them in so they ride the same encoder walk.
        fn = getattr(self, "recon_loss_fn", None)
        if fn is not None and hasattr(fn, "training_metrics"):
            try:
                out.update(fn.training_metrics())
            except Exception:
                pass
        return out

    def dashboard_snapshots(self) -> Dict[str, dict]:
        """Non-scalar snapshots from the reconstruction loss (e.g. HALO's
        energy ring), surfaced via the encoder."""
        fn = getattr(self, "recon_loss_fn", None)
        if fn is not None and hasattr(fn, "dashboard_snapshots"):
            try:
                return fn.dashboard_snapshots() or {}
            except Exception:
                return {}
        return {}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _patch_vote_sample(
        self,
        h_last: torch.Tensor,
        temperature: float,
        num_samples: int = 200,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Paper Algorithm 1 (temperature sampling for CALM).

        ``temperature`` must be in (0, 1]; for T<1 we draw ``num_samples``
        candidate latents, decode each to an argmax K-token patch, count
        exact patch matches, and pick the most-supported patch via
        combinatorial weighting (cascade from n_initial = round(1/T) down).
        For T=1 a single latent is sampled - matching the reference. This is
        the paper's count-based temperature; the energy head's noise is full
        (``noise_scale=1.0``) as in the reference.

        Args:
            h_last: ``[hidden_size]`` conditioning (single stream).
            temperature: T in (0, 1]. Strict T = 1/integer in the paper;
                we round 1/T to the nearest integer so any T in range works.
            num_samples: pool size for voting.
            noise_scale: NON-paper diagnostic knob. <1 shrinks the head's
                noise toward its conditional-mean (best-guess) prediction;
                useful to peek at a weakly-trained head, but it bypasses the
                rigorous count-based temperature. Default 1.0 = paper-faithful.

        Returns:
            ``[K]`` selected token ids.
        """
        if temperature >= 1.0:
            z = self.energy_head.sample(h_last, num_samples=1, noise_scale=noise_scale)
            z = z.view(1, 1, -1)
            recon_hidden = self.vae.decode(z)
            recon_logits = self._classify(recon_hidden)
            return recon_logits.argmax(dim=-1).view(self.K)

        n_initial = max(1, int(round(1.0 / max(temperature, 1e-6))))

        # Pool of candidate patches: decode each candidate latent and take
        # argmax tokens. Stochasticity lives in the latent draws; voting then
        # concentrates on the patches the head agrees on most often (paper
        # Algorithm 2). noise_scale stays 1.0 for paper-faithful sampling.
        z_pool = self.energy_head.sample(
            h_last, num_samples=num_samples, noise_scale=noise_scale
        )
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
                chosen = random.choices(list(candidates.keys()), weights=weights, k=1)[
                    0
                ]
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
        # Non-paper diagnostic: <1 shrinks the head's noise toward its mean
        # prediction (peek at a weakly-trained head). Default 1.0 = paper-faithful.
        noise_scale = float(getattr(generation_config, "calm_noise_scale", None) or 1.0)
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
                h_last[0],
                temperature=float(temperature),
                num_samples=num_samples,
                noise_scale=noise_scale,
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
