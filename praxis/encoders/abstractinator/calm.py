"""AbstractinatorCALM: a continuous CALM arm beside the discrete RVQ arm.

WHY THIS EXISTS. CALM (arXiv 2510.27688) autoregresses over *continuous*
latents and resolves each step by a count-based vote over many draws rather
than by an argmax over a categorical - temperature realized as a draw count,
``n = round(1/T)``. That is the mechanism worth having. Its cost is that the
energy head must learn a full continuous conditional ``p(z_{t+1} | context)``
from an energy score, which is a weak, high-variance signal: measured here at
correct-next-patch 0.003 and cross-sample agreement 0.000. Not a bug - the port
is faithful and the architecture is correct - but an optimization that needs
far more tokens than this line can afford.

The Abstractinator has exactly what that objective lacks: a DISCRETE code per
patch, supervised end-to-end by cross-entropy through the byte decoder, and
therefore dense, low-variance and mode-seeking. So this encoder runs both and
lets the discrete arm pay for the continuous one:

    patch features (already in CALM's standing-wave basis)
        |
        +--> HarmonicResidualVQ ------> z_q      discrete, reconstruction-sufficient
        |
        +--> Gaussian posterior ------> z_c      continuous, carries what
        |                                        quantization threw away
        v
    trunk input  z = z_q + z_c
        |
    global transformer
        |
        +--> EnergyHead: propose z_{t+1}         CALM's generative path
        +--> code CE:    predict code_{t+1}      the dense signal that
                                                 concentrates the conditional

WHY THE CODE CE AND NOT THE EXISTING ANCHOR. ``CALMEncoder`` already carries an
``ENERGY_ANCHOR_WEIGHT`` MSE from the head's zero-noise prediction onto the next
posterior MEAN, and it works (correct-next-patch 0.003 -> 0.139). But an MSE
onto a mean is MEAN-SEEKING, which is precisely the blur CALM's energy score
exists to avoid - the code says so, and calls it a deviation from the paper.
Cross-entropy over codebook entries is MODE-SEEKING and carries no such defect.
It is also a better target than the trinary mode's HALO cells: those partition a
frozen codec after the fact, while this codebook is trained jointly with EMA
updates and dead-code revival, and is guaranteed sufficient to reconstruct.

JOINT, SINGLE-STAGE, ALWAYS. No pretraining phase, no frozen codec, no stage
boundary - the standing requirement on this line. Everything above folds into
one loss container inside one training loop. This is also the honest risk: the
CALM reference gets its clean conditional partly BECAUSE its codec is frozen and
near-lossless before the energy head starts, and here the head chases a moving
target. The bet is that the code CE is dense enough to make that tractable, and
``calm_code_acc`` is where the bet is settled.

THE VOTE, AND WHY IT IS CHEAPER HERE. CALM votes by decoding every candidate to
a K-token patch and counting exact matches - a full decode per draw. Here the
RVQ already defines the equivalence classes: quantizing a candidate latent
yields its code index directly, so the vote is a nearest-neighbour lookup per
draw instead of a decode. Same count-based vote, a fraction of the cost, and the
discrete arm is what makes it possible.
"""

import math
from typing import Any, Dict, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.encoders.abstractinator.encoder import AbstractinatorEncoder
from praxis.encoders.calm.vae import PatchVAE
from praxis.heads.energy import EnergyHead
from praxis.losses.energy_score import energy_score_loss
from praxis.losses.uncertainty import UncertaintyWeighting

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# Draws per vote at generation. CALM's own value, and the point of the
# construction: temperature is realized as a COUNT (n = round(1/T)) rather than
# as a logit scale, so this is a structural number, not a tuned one.
VOTE_SAMPLES: int = 500

# Default vote temperature. This is a GENERATION knob, not a training
# hyperparameter: the reference realizes temperature as the cascade's starting
# count, ``n = round(1/T)``, so T=0.5 asks for a cell with at least two votes
# before it will settle for one. T=1 is a single draw with no vote at all.
VOTE_TEMPERATURE: float = 0.5

# Draws for the energy score's model term (N) and posterior term (M) during
# training. Both are the reference's: `config.num_samples = 8` for the model
# term, and `n_y = 100` hardcoded inside `EnergyTransformer.energy_score`. M was
# 16 here, which is 6x more variance on the attractive term - and that variance
# goes straight into the trunk. The target draws cost no head forward (they are
# `mean + eps * std`), only the N x M distance matrix, so the reference's number
# is affordable.
ENERGY_SAMPLES_N: int = 8
ENERGY_SAMPLES_M: int = 100

# Free-bits floor for the continuous arm's KL, in nats per latent dimension.
# Below this the KL is not penalized, which stops the posterior collapsing onto
# the prior in the early steps when the discrete arm alone can explain the data
# - the failure this arm exists to avoid. The reference's `kl_clamp`, which was
# 0.05 here: a 10x looser floor than the codec CALM actually trains against.
FREE_BITS: float = 0.5

# Codec dropout, the reference's `ae_dropout`. NOT a regularization knob: it is
# what makes the decoder tolerant of a latent the energy head PREDICTED rather
# than one the encoder produced, which is the only train/test gap CALM's design
# actually has. Applied to the encoder's input features and to the sampled
# latent before reconstruction, so the decoder learns to map a NEIGHBOURHOOD of
# z to the right features. `CALMVAE` already carried it at this value and calls
# both sites load-bearing for generation; `PatchVAE` was first built with it at
# 0, which silently removed the reference's own answer to the problem.
VAE_DROPOUT: float = 0.15

# NO HAND-SET WEIGHTS. `CODE_CE_WEIGHT = 5.0` and `KL_WEIGHT = 1e-3` used to
# live here and are gone: the three CALM objectives are now balanced by learned
# uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018 - see
# praxis/losses/uncertainty.py).
#
# WHY, measured on run b32ddef0f. With the arm capped to 0.25% of the discrete
# one - i.e. contributing essentially nothing to the trunk's input - the run
# still degraded on every axis that matters: all three prismatic arms got WORSE
# (`arm_solo_loss_1` +3.4/1k, `_2` +1.4/1k, `_0` +0.5/1k), `halting/eval_mean_kl`
# rose 10x, `vq_resets_s0` doubled to 2178, and `val_byte_nll_bits` went
# 5.053 -> 5.43. The arm was silent, so the damage was not the arm: it was the
# CALM LOSSES' GRADIENT arriving on the trunk. `calm_energy` sits near 16 at
# init (2*sqrt(D) against unit-RMS targets) while the byte CE is around 5, so
# the hard, slow objective was outweighing the main task roughly 3:1 from step
# zero - and `calm_energy_cond_gap` says it was not even using its conditioning
# while it did so.
#
# Uncertainty weighting settles each objective's weight at 1/L, so an objective
# that stays hard down-weights ITSELF and cannot drown a task that is already
# working, then re-engages on its own as it becomes learnable. That is the
# balance this arm needed, it is learned rather than tuned, and it removes two
# constants instead of adding any.

# Hard ceiling on ||z_c|| / ||z_q||, enforced per patch.
#
# THE FAILURE THIS EXISTS TO PREVENT, measured on -p run b32ddef0f. The arm was
# added as a bare residual, `z = z_q + z_c`, with only the 1e-3 KL opposing its
# growth. `calm_arm_ratio` went 7e-4 -> 1.5 by step 500 and 20-35 by step 2500:
# the continuous channel became 95%+ of the trunk's input. Everything else
# followed from that one number:
#
#   - The byte decoder's CE had almost no reason to differentiate CODES, so the
#     codebook starved: `vq_dead_frac_s0` 0.73, `vq_perplexity_s0` 6-10 of K,
#     1200+ dead-code resets.
#   - Those resets relabel the code CE's targets under it, so `calm_code_acc`
#     slammed between ~0.6 and EXACTLY chance (1/K) batch to batch.
#   - `calm_energy` and the KL both scale with ||z||, so train `loss` sat at
#     330-650 with spikes to 5572, and `val_byte_nll_bits` ROSE, 5.169 -> 5.355.
#
# WHY THE KL COULD NEVER HAVE HELD IT. Every objective whose cost grows with
# ||z_c|| is detached from the posterior by design: the energy score's target is
# `z.detach()` and its target draws use `mu.detach()` (correct - the score must
# train the head, not the posterior). So the one term that both sees the
# magnitude and can act on it is the KL, at 1e-3 against a fully-connected
# decoder CE that prefers an unbottlenecked channel. Raising that weight would
# be a tuned number opposing an untuned one, and would mute the arm rather than
# bound it.
#
# So the bound is STRUCTURAL, and parity is the principled place for it: at
# ratio 1 the continuous arm has stopped being a residual on the discrete one
# and this is no longer the architecture under test. The scaling is a SOFT cap -
# below the ceiling z_c passes through untouched, so the silent start and the
# KL's shaping both survive - and the fraction of the ceiling actually used is
# LEARNED (`arm_gate`), so nothing here is a tuned level.
ARM_CEILING: float = 1.0

# Logit init for that learned fraction. sigmoid(-6) ~ 0.0025, so the arm still
# starts effectively silent and -p remains an A/B against -o rather than a
# reroll. The cap is soft, so this does not scale a near-zero z_c UP to it.
ARM_GATE_INIT: float = -6.0


class AbstractinatorCALM(AbstractinatorEncoder):
    """Abstractinator with a continuous CALM codec and a count-based vote.

    Everything the parent does is unchanged; this adds a VAE beside the
    quantizer, an energy head over the trunk output, and the dense code
    objective that pays for it.

    BOTH DECODING PATHS ARE TRAINED AT ONCE, so which one a run generates with
    is a runtime choice and not part of the model's identity: `--generation-mode`
    is excluded from the run hash, and the same checkpoint can be read out
    either way. That is the whole reason this is not two registry profiles -
    two profiles would mean two training runs just to compare decoders.

    "standard" is the default: byte-at-a-time through the byte decoder's
    logits, and MTP speculative decode where it applies. "vote" hands
    generation to CALM's patch vote. The two loops are mutually exclusive -
    you autoregress over bytes or over patches, never both.
    """

    generation_modes = ("standard", "vote")
    default_generation_mode = "standard"

    def __init__(
        self,
        config: ConfigType,
        *,
        energy_hidden_ratio: float = 1.0,
        energy_blocks: int = 3,
        vae_depth: int = 2,
        vote_samples: int = VOTE_SAMPLES,
        vote_temperature: float = VOTE_TEMPERATURE,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        D = config.hidden_size
        self.vote_samples = int(vote_samples)
        self.vote_temperature = float(vote_temperature)

        # The continuous arm: a Gaussian posterior over the SAME patch features
        # the quantizer sees. Deliberately not CALM's token-chunk VAE - that
        # compresses K token embeddings and owns its own table, which is a job
        # the Abstractinator's local encoder already did. What is needed here is
        # only the continuous channel, so this is the posterior and nothing else.
        # The byte decoder supplies the reconstruction pressure, so no second
        # decoder exists to disagree with the one already there.
        # THE CONTINUOUS CODEC. A real VAE - encoder, decoder, its own
        # reconstruction objective, its own free-bits KL - over the same patch
        # features the quantizer sees. This replaces a bare
        # `nn.Linear(D, 2 * D)` "posterior", and that substitution is what four
        # rounds of instability were actually about.
        #
        # The original reasoning for dropping CALM's VAE checked exactly one of
        # the three jobs it does. It compresses K tokens into one vector - and
        # the Abstractinator's local encoder genuinely already did that, so a
        # second token-chunk VAE really would be redundant. But it also supplies
        # a continuous, KL-regularized, unit-scale, STATIONARY latent space, and
        # a per-patch POSTERIOR for the energy score's target draws. Neither is
        # redundant with an RVQ, and both were dropped along with the
        # compression. They came back one crisis at a time: as a scale runaway
        # (band-aided with an RMS map at the loss), and as a linear layer with
        # no reconstruction pressure of its own, which had to be capped to 0.25%
        # because it destabilized everything it touched.
        #
        # Two encoders side by side, sharing one trunk. That is well-posed here
        # ONLY because the patching is static (`patch_size=8`): both codecs emit
        # exactly one latent per patch, so `z_q + z_c` is an alignable merge and
        # not two sequences that cannot be reconciled. Each path carries its own
        # objective, so each is independently optimizable - which is the point.
        self.vae = PatchVAE(
            feature_dim=D,
            latent_dim=D,
            hidden_dim=D,
            depth=vae_depth,
            latent_norm=True,
            dropout=VAE_DROPOUT,
        )

        hidden = max(1, int(D * energy_hidden_ratio))
        # `noise_dim` is the reference's `noise_size = 64` against
        # `latent_size = 128` - the noise is deliberately NARROWER than the
        # latent it has to cover, so the head is forced to use the conditioning
        # to fill the rest. It was D here, i.e. noise as wide as the target,
        # which lets the head satisfy the score from noise alone.
        self.energy_head = EnergyHead(
            cond_dim=D,
            noise_dim=max(1, D // 2),
            latent_dim=D,
            hidden_dim=hidden,
            num_blocks=energy_blocks,
        )
        # Predicts WHICH CODE comes next from the same conditioning hidden the
        # energy head reads, so its gradient sharpens the very representation
        # the head conditions on. One head per residual stage: the composed
        # index is a product space and a single softmax over K**depth would be
        # both huge and badly conditioned.
        core = getattr(self.quantizer, "quantizer", self.quantizer)
        self.vq_depth = int(getattr(core, "depth", 1))
        self.vq_K = int(getattr(core, "K", 0))
        self.code_heads = nn.ModuleList(
            [nn.Linear(D, self.vq_K) for _ in range(self.vq_depth)]
        )

        # Learned fraction of ARM_CEILING the continuous arm may occupy. Shape
        # [1] rather than 0-dim: schedule_free's swap() views parameters as
        # uint8 and 0-dim parameters break it.
        self.arm_gate = nn.Parameter(torch.full((1,), ARM_GATE_INIT))

        # Learned balance across the three CALM objectives. Every weight starts
        # at exactly 1, so step 0 is unchanged and this stays an A/B.
        self.loss_balance = UncertaintyWeighting(
            ("calm_energy", "calm_code_ce", "calm_kl", "calm_recon")
        )

        self._pending: Dict[str, torch.Tensor] = {}
        self._calm_diag: Dict[str, float] = {}
        self._last_latent: Optional[torch.Tensor] = None
        self._last_code: Optional[torch.Tensor] = None
        self._last_code_mean: Optional[torch.Tensor] = None
        self._last_posterior: Optional[tuple] = None
        self._last_stage_indices: Optional[list] = None

    # ── training ───────────────────────────────────────────────────────────

    def _post_downsample(self, h, aux_loss):
        """Parent's RVQ, plus the continuous codec added residually."""
        z_q, aux = super()._post_downsample(h, aux_loss)

        # ── the continuous codec, on its own objectives ────────────────────
        mu, logvar = self.vae.encode(h)
        z_c = self.vae.reparameterize(mu, logvar) if self.training else mu
        self._last_posterior = (mu, logvar)

        if self.training:
            # ITS OWN RECONSTRUCTION. This is the job the bare linear could not
            # do: nothing else in the model requires the continuous latent to
            # be informative, so without this the KL simply wins and the arm is
            # noise with a gate on it.
            recon = self.vae.reconstruction_loss(self.vae.decode(z_c), h)
            self._pending["calm_recon"] = self.loss_balance("calm_recon", recon)
            self._calm_diag["calm_recon_rel"] = float(recon.detach())

            kl = self.vae.kl_divergence(mu, logvar, per_dim_clip=FREE_BITS)
            self._pending["calm_kl"] = self.loss_balance("calm_kl", kl)

        # The latent the energy head predicts, in the VAE's OWN geometry. Unit
        # per-dim RMS by the codec's contract, so the target space is stationary
        # by construction - not by an RMS map applied at the loss to correct for
        # an unbounded quantizer output, which is what this was before.
        self._last_code = self.vae.normalize_latent(z_c)
        self._last_code_mean = self.vae.normalize_latent(mu).detach()

        # SOFT CAP on what the continuous codec contributes to the trunk. Below
        # the ceiling z_c is untouched; above it, projected back. `z_q` is
        # detached so the model cannot widen its allowance by shrinking the
        # discrete arm. The gate is learned, so `calm_arm_gate` rising is now
        # meaningful evidence the arm earned its place - it has a real objective
        # behind it for the first time.
        ceiling = ARM_CEILING * torch.sigmoid(self.arm_gate)
        cap = ceiling * z_q.detach().norm(dim=-1, keepdim=True)
        contribution = self._last_code
        contribution = contribution * (
            cap / contribution.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ).clamp(max=1.0)

        z = z_q + contribution
        self._last_latent = z
        self._calm_diag["calm_arm_ratio"] = float(
            contribution.detach().norm() / z_q.detach().norm().clamp_min(1e-6)
        )
        self._calm_diag["calm_arm_gate"] = float(ceiling.detach())
        return z, aux

    def _stage_indices(self) -> Optional[list]:
        """Per-stage code ids for the last forward, if the bank exposes them."""
        core = getattr(self.quantizer, "quantizer", self.quantizer)
        codec = getattr(core, "codec", None)
        idx = getattr(self, "_last_vq_indices", None)
        if codec is None or idx is None:
            return None
        digits, _ = codec.decompose(idx)
        return digits

    def decode(self, h, *args: Any, **kwargs: Any):
        """Register the CALM-side objectives, then decode as the parent does.

        ``h`` is the trunk output over patches, so position ``p`` here is the
        conditioning for patch ``p+1`` - the same alignment CALM uses.
        """
        if self.training and torch.is_grad_enabled() and h.shape[1] >= 2:
            self._register_calm_losses(h)
        return super().decode(h, *args, **kwargs)

    def _register_calm_losses(self, h: torch.Tensor) -> None:
        h_cond = h[:, :-1, :]  # conditioning for positions 1..N-1

        # ── the dense arm: which CODE comes next ───────────────────────────
        digits = self._stage_indices()
        if digits is not None:
            ce = h.new_zeros(())
            acc = 0.0
            for s, head in enumerate(self.code_heads):
                if s >= len(digits):
                    break
                target = digits[s][:, 1:].reshape(-1)
                logits = head(h_cond).reshape(-1, self.vq_K)
                ce = ce + F.cross_entropy(logits.float(), target)
                acc += float((logits.detach().argmax(-1) == target).float().mean())
            n = max(1, min(len(digits), len(self.code_heads)))
            # Normalize by the chance level so the term is dimensionless: 1.0 is
            # chance, 0 is perfect. Without this the objective's magnitude was
            # an accident of codebook size - at K=512 a head sitting at chance
            # parked ~31 nats in the total loss forever, and pushed 5x-amplified
            # noise into the trunk while the codebook was thrashing.
            chance = math.log(max(2, self.vq_K))
            self._pending["calm_code_ce"] = self.loss_balance(
                "calm_code_ce", (ce / n) / chance
            )
            self._calm_diag["calm_code_acc"] = acc / n

        # ── the continuous arm: CALM's energy score ────────────────────────
        # The head now predicts the next VAE LATENT, which is what CALM's head
        # predicts. That target is KL-regularized and unit per-dim RMS by the
        # codec's own contract, so the space is stationary by construction. It
        # used to be `z_q + z_c` - the raw quantizer output, unnormalized and
        # reshaped by the same gradient step - with an RMS map bolted on at the
        # loss to hide the consequences.
        code = self._last_code
        if code is None or self._last_posterior is None:
            return
        B, P, D = code[:, 1:, :].shape
        proposals = self.energy_head.sample(
            h_cond, num_samples=ENERGY_SAMPLES_N
        ).permute(1, 2, 0, 3)

        # Target draws from the NEXT patch's posterior - the reference's
        # `mean + eps * std`, at its `n_y = 100`. Centred on the MEAN, never on
        # a draw: `z_c` already carries one sample of that noise, so centring
        # there applied it twice around a centre that moved every step.
        _, logvar = self._last_posterior
        centre = self._last_code_mean[:, 1:, :].unsqueeze(2)
        std_n = (0.5 * logvar[:, 1:, :]).exp().detach().unsqueeze(2)
        targets = (
            centre
            + torch.randn(
                B, P, ENERGY_SAMPLES_M, D, device=h.device, dtype=centre.dtype
            )
            * std_n
        )
        loss = energy_score_loss(proposals, targets)
        self._pending["calm_energy"] = self.loss_balance("calm_energy", loss)

        with torch.no_grad():
            # Is the head USING the conditioning, or has it collapsed onto the
            # marginal? Re-score the same draws against targets rolled one
            # position out of alignment. Near 0 means it is modelling the
            # marginal and charging the trunk for it.
            if P > 2:
                mismatched = energy_score_loss(
                    proposals.detach(), targets.detach().roll(1, dims=1)
                )
                self._calm_diag["calm_energy_cond_gap"] = float(mismatched - loss)

            # Does the head CONCENTRATE? Decode each proposal through the VAE
            # and quantize the RECONSTRUCTED FEATURE, which is the space the
            # codebook actually indexes - `_quantize_to_codes` applies the
            # analysis rotation, and that is defined on patch features, not on
            # latents. Feeding it raw latents was quantizing in the wrong space.
            recon = self.vae.decode(proposals.reshape(-1, D))
            codes = self._quantize_to_codes(recon).view(B, P, ENERGY_SAMPLES_N)
            if ENERGY_SAMPLES_N > 1:
                agree = (codes[..., :1] == codes[..., 1:]).float().mean()
                self._calm_diag["calm_sample_agreement"] = float(agree)

    @torch.no_grad()
    def _quantize_to_codes(self, z: torch.Tensor) -> torch.Tensor:
        """Composed RVQ index for arbitrary latents ``[N, D]``, READ-ONLY.

        The vote's equivalence classes. CALM has to decode each candidate to a
        K-token patch to compare them; the codebook already defines that
        partition, so one nearest-neighbour lookup replaces a whole decode.

        This does the residual staging by hand rather than calling the
        quantizer's ``forward``, and that is not a micro-optimization. The
        quantizer's forward MUTATES: it writes the replacement buffer, runs EMA
        codebook updates and fires dead-code resets. Voting through it would
        push 500 candidate latents per generated patch into the live codebook -
        caught by a test asserting that quantizing the same tensor twice gives
        the same answer, which it did not: the second call collapsed seven
        distinct latents onto a single code.
        """
        core = getattr(self.quantizer, "quantizer", self.quantizer)
        analysis = getattr(self.quantizer, "analysis", None)
        x = z if analysis is None else z @ analysis
        if analysis is not None:
            gdn = getattr(self.quantizer, "gdn", None)
            x = (
                gdn(x)
                if gdn is not None
                else x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
            )
        residual = x
        digits = []
        for stage in range(self.vq_depth):
            book = core.stage_codebook(stage)  # [K, L]
            d = (
                residual.pow(2).sum(-1, keepdim=True)
                - 2.0 * residual @ book.T
                + book.pow(2).sum(-1).unsqueeze(0)
            )
            idx = d.argmin(-1)
            digits.append(idx)
            residual = residual - F.embedding(idx, book)
        if hasattr(core, "_compose_indices"):
            return core._compose_indices(digits).reshape(-1)
        return digits[0]

    def consume_pending_losses(self) -> Dict[str, torch.Tensor]:
        out, self._pending = self._pending, {}
        return out

    def training_metrics(self) -> dict:
        out = super().training_metrics()
        out.update(self._calm_diag)
        for name, w in self.loss_balance.weights().items():
            out[f"calm_weight_{name.removeprefix('calm_')}"] = w
        return out

    # ── generation: the count-based vote ───────────────────────────────────

    @torch.no_grad()
    def vote_next_latent(
        self, h_cond: torch.Tensor, temperature: float = VOTE_TEMPERATURE
    ) -> torch.Tensor:
        """One CALM vote step, in code space. ``h_cond``: ``[B, D]``.

        Faithful to ``CALMEncoder._patch_vote_sample`` (the authors'
        ``temperature_sampling``) in the two places it is easy to get wrong:

        SELECTION IS NOT ARGMAX. The reference cascades ``n`` from
        ``round(1/T)`` down to 1, keeps the cells with at least ``n`` votes, and
        makes a RANDOM choice among them weighted by ``C(count, n)``. That
        combinatorial weighting IS the temperature - a plain modal argmax is
        this algorithm's ``T -> 0`` limit and nothing else, which silently
        deletes the only sampling control the design has.

        AND NOTHING IS AVERAGED. The reference votes in TOKEN space, which is
        its output space, so the winner is simply emitted. Here the vote selects
        a cell but the trunk still needs a vector, and the tempting move -
        averaging the proposals inside the winning cell - reintroduces exactly
        the conditional-mean estimator the energy score exists to avoid: if the
        codebook is coarse relative to the conditional's spread, most proposals
        land in one cell and that average IS the global mean. So the winner is a
        genuine draw uniformly from the cell's members, never a synthetic point.

        ``calm_vote_lift`` is the receipt on that: it measures how far the
        returned latent sits from the mean of ALL proposals, in units of their
        spread. Near 0 means the vote is decorative whatever its margin says.
        """
        B, _ = h_cond.shape
        n_draws = self.vote_samples
        # Through the head's own sampler, so generation draws the same uniform
        # [-0.5, 0.5] noise training does - and so the noise width stays the
        # head's, not the latent's. [n, B, D] -> [B, n, D].
        proposals = self.energy_head.sample(h_cond, num_samples=n_draws).permute(
            1, 0, 2
        )
        D = proposals.shape[-1]
        # The reference votes by DECODING every candidate and comparing the
        # tokens. Same thing here: decode the latent through the VAE, then read
        # off the RVQ cell of the reconstructed feature - one nearest-neighbour
        # lookup instead of a full byte decode, which is what having the
        # discrete codec beside the continuous one buys.
        codes = self._quantize_to_codes(self.vae.decode(proposals.reshape(-1, D))).view(
            B, n_draws
        )

        # Count-based temperature: n = round(1/T), exactly as the reference.
        n_initial = max(1, int(round(1.0 / max(float(temperature), 1e-6))))

        out = proposals.new_empty(B, D)
        margins, lifts = [], []
        for b in range(B):
            vals, counts = torch.unique(codes[b], return_counts=True)
            margins.append(float(counts.max()) / n_draws)
            winner = self._cascade_pick(vals, counts, n_initial)
            members = proposals[b][codes[b] == winner]
            # A uniform draw from the winning cell - a real sample, not a mean.
            out[b] = members[torch.randint(len(members), (1,), device=members.device)][
                0
            ]
            if n_draws > 1:
                spread = proposals[b].std(0).mean().clamp_min(1e-6)
                lifts.append(
                    float((out[b] - proposals[b].mean(0)).norm() / (spread * D**0.5))
                )
        self._calm_diag["calm_vote_margin"] = sum(margins) / max(1, len(margins))
        if lifts:
            self._calm_diag["calm_vote_lift"] = sum(lifts) / len(lifts)
        return out

    @staticmethod
    def _cascade_pick(
        vals: torch.Tensor, counts: torch.Tensor, n_initial: int
    ) -> torch.Tensor:
        """The reference's cascade: descend ``n``, keep cells with >= n votes,
        choose among them weighted by ``C(count, n)``.

        Higher ``n`` (lower temperature) demands more agreement before a cell is
        eligible, and the binomial weight sharpens toward the best-supported one
        without ever collapsing to a hard argmax. At ``n = 1`` every observed
        cell is eligible weighted by its raw count, which is ordinary sampling.
        """
        for n in range(n_initial, 0, -1):
            keep = counts >= n
            if not bool(keep.any()):
                continue
            c = counts[keep].to(torch.float64)
            # C(c, n) in log space: lgamma(c+1) - lgamma(n+1) - lgamma(c-n+1).
            logw = torch.lgamma(c + 1) - math.lgamma(n + 1) - torch.lgamma(c - n + 1)
            probs = torch.softmax(logw - logw.max(), dim=0)
            return vals[keep][int(torch.multinomial(probs, 1).item())]
        return vals[int(counts.argmax())]

    def _trunk_input(self, h_hat: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """Trunk input for a PREDICTED patch: ``z_q + capped z_c``.

        The training path builds this from a real patch feature; here the
        feature is the VAE's decode of a voted latent, so the discrete half is
        that feature's own quantization. Same cap as training, so a patch the
        model predicts enters the trunk on exactly the terms a patch it
        observed does.
        """
        # The quantizer's forward MUTATES - EMA codebook updates and dead-code
        # resets - but both are gated on `self.training`, and generation only
        # runs in eval, so this read is safe. Asserted rather than assumed:
        # pushing generated candidates into the live codebook is the exact bug
        # `_quantize_to_codes` was hand-rolled to avoid.
        assert not self.training, "vote generation must run in eval"
        z_q = self.quantizer(h_hat.unsqueeze(1))[0].squeeze(1)
        ceiling = ARM_CEILING * torch.sigmoid(self.arm_gate)
        cap = ceiling * z_q.detach().norm(dim=-1, keepdim=True)
        contribution = z_hat * (
            cap / z_hat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ).clamp(max=1.0)
        return z_q + contribution

    @torch.no_grad()
    def custom_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        base_forward,
        generation_config=None,
        latent_forward=None,
        decode_logits=None,
        **kwargs: Any,
    ):
        """CALM's vote, driving generation one PATCH at a time.

        Returns ``None`` - deferring to the standard byte loop - unless the run
        resolved to ``generation_mode="vote"``, so this is selectable per run
        rather than replacing byte-level generation outright.

        Each patch costs ONE trunk forward, which is the whole point:

          1. run the trunk over the bytes so far;
          2. VOTE the next latent from the last patch's hidden - draw
             ``vote_samples`` proposals from the energy head, quantize each
             through the VAE decoder to get its RVQ cell, and pick by the
             reference's C(count, n) cascade;
          3. decode the winner to a patch feature, put it back on the trunk as
             a predicted patch (this is what ``latent_forward`` is for - the
             patch has no bytes behind it, so ``base_forward`` cannot reach it);
          4. emit that patch's bytes from the local byte decoder, conditioned on
             the trunk output for the predicted patch.

        Step 4 is byte-at-a-time, but the local decoder is causal, so a byte
        reads only its predecessors and placeholder tail bytes cannot
        contaminate it. Static patching is what makes the whole loop well posed:
        boundaries are at fixed multiples of ``patch_size``, so the patch the
        vote is predicting is known before any of its bytes exist.
        """
        if (
            self.generation_mode != "vote"
            or inputs is None
            or latent_forward is None
            or decode_logits is None
        ):
            return None

        from transformers import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        max_new = getattr(generation_config, "max_new_tokens", 100) or 100
        # The VOTE's temperature is a draw COUNT (n = round(1/T)), a different
        # quantity from the logit temperature above; `calm_vote_temperature` on
        # the generation config overrides it, never the sampler's `temperature`.
        vote_t = float(
            getattr(generation_config, "calm_vote_temperature", None)
            or self.vote_temperature
        )
        K = int(self.byte_config.patch_size)
        # `return_dict_in_generate` arrives as a KWARG on model.generate (see
        # DecodeBackend), not on the config, and the caller then reads
        # `.sequences`. Returning a bare tensor made every queued generation
        # fail with "'Tensor' object has no attribute 'sequences'".
        return_dict = bool(
            kwargs.get("return_dict_in_generate")
            or getattr(generation_config, "return_dict_in_generate", False)
        )
        # DEADLINE. Queued generations decode inside the training loop, so a
        # loop that ignores the caller's stopping criteria is a stalled run.
        # transformers never runs the criteria list for a loop it does not own,
        # which is why the speculative path checks them by hand too.
        stopping_criteria = kwargs.get("stopping_criteria")
        # SAMPLING. This loop owns its decoding, so transformers builds no
        # processor list for it and every knob has to be honored by hand - the
        # same reason the speculative path does it explicitly.
        #
        # Hard-coding argmax here made two "independent" draws BYTE-IDENTICAL,
        # which silently destroyed BrierLM: its estimator is
        # `1{a=y} + 1{b=y} - 1{a=b}` over two i.i.d. samples, so a=b forces the
        # self-match term to 1 and every order non-positive. Floored, the
        # geometric mean is then exactly 0 by construction - the metric could
        # not report anything else, whatever the model had learned. The vote's
        # own randomness did not rescue it: the continuous arm enters the trunk
        # at a fraction of a percent, so a differently-voted patch barely moves
        # the byte decoder's argmax.
        do_sample = bool(getattr(generation_config, "do_sample", False))
        temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)
        rep_penalty = float(getattr(generation_config, "repetition_penalty", 1.0) or 1.0)
        top_k = getattr(generation_config, "top_k", None)
        top_p = getattr(generation_config, "top_p", None)
        penalizers = LogitsProcessorList()
        if rep_penalty != 1.0:
            penalizers.append(RepetitionPenaltyLogitsProcessor(penalty=rep_penalty))
        warpers = LogitsProcessorList()
        if do_sample:
            if top_k:
                warpers.append(TopKLogitsWarper(int(top_k)))
            if top_p is not None and top_p < 1.0:
                warpers.append(TopPLogitsWarper(float(top_p)))

        def pick(raw_logits, context_ids):
            scores = penalizers(context_ids, raw_logits)
            if do_sample and temperature > 0:
                scores = warpers(context_ids, scores)
                probs = F.softmax(scores / temperature, dim=-1)
                return torch.multinomial(probs, 1)
            return scores.argmax(dim=-1, keepdim=True)

        eos_id = getattr(generation_config, "eos_token_id", None)
        eos = (
            {eos_id}
            if isinstance(eos_id, int)
            else set(eos_id) if isinstance(eos_id, (list, tuple)) else set()
        )

        generated = inputs
        produced = 0
        while produced < max_new:
            out = base_forward(generated)
            z_hat = self.vote_next_latent(
                out.last_hidden_state[:, -1, :], temperature=vote_t
            )
            h_hat = self.vae.decode(z_hat)
            z_next = self._trunk_input(h_hat, z_hat).unsqueeze(1)
            z_ext = torch.cat([out.patch_embeds, z_next], dim=1)
            h_ext = latent_forward(z_ext)

            stop = False
            for _ in range(min(K, max_new - produced)):
                pos = generated.shape[1]
                pad = generated.new_zeros((generated.shape[0], K))
                ext = torch.cat([generated, pad], dim=1)
                _, h_enc, plens, blk, _, ltoks = self.encode(ext)
                logits, embeds = self.decode(
                    h_ext[:, : plens.shape[1]], h_enc, ext, plens, ltoks, blk
                )
                if logits is None:
                    logits = decode_logits(embeds)
                nxt = pick(logits[:, pos - 1, :], generated)
                generated = torch.cat([generated, nxt], dim=1)
                produced += 1
                if eos and int(nxt.view(-1)[0]) in eos:
                    stop = True
                    break
                if stopping_criteria is not None and bool(
                    stopping_criteria(generated, None).all()
                ):
                    stop = True
                    break
            if stop:
                break
        if return_dict:
            from types import SimpleNamespace

            return SimpleNamespace(sequences=generated)
        return generated

    metric_descriptions = {
        **AbstractinatorEncoder.metric_descriptions,
        "calm_code_acc": {
            "description": (
                "Next-patch RVQ code accuracy from the conditioning hidden - "
                "the dense signal paying for the energy head. Climbing WITH "
                "calm_sample_agreement is the thesis; alone means decoupled."
            ),
            "chart": {
                "title": "CALM Code Accuracy",
                "y_label": "next-code accuracy",
                "group": "calm_arm",
                "group_order": 480,
                "order": 10,
            },
        },
        "calm_sample_agreement": {
            "description": (
                "Share of energy-head draws landing on the first draw's code. "
                "Expect a U: the 1.0 at step 0 is degenerate; only the rise "
                "after the fall counts. Sustained ~0 = marginal collapse."
            ),
            "chart": {
                "title": "CALM Sample Agreement",
                "y_label": "cross-draw code agreement",
                "group": "calm_arm",
                "order": 20,
            },
        },
        "calm_arm_ratio": {
            "description": (
                "||z_c|| / ||z_q||: the continuous arm beside the quantized "
                "one. Hard-capped at ARM_CEILING; above ~1 it stopped being a "
                "residual and starves the codebook, which is what broke -p."
            ),
            "chart": {
                "title": "CALM Arm Ratio",
                "y_label": "||z_c|| / ||z_q||",
                "group": "calm_arm",
                "order": 30,
            },
        },
        **{
            f"calm_weight_{short}": {
                "description": (
                    f"Learned weight on {label} (Kendall et al. uncertainty "
                    "weighting). Settles at 1/loss, so an objective that stays "
                    "hard down-weights itself instead of drowning the trunk."
                ),
                "chart": {
                    "title": f"CALM Weight: {title}",
                    "y_label": "exp(-log var)",
                    "group": "calm_arm",
                    "order": order,
                },
            }
            for short, label, title, order in (
                ("energy", "the energy score", "Energy", 60),
                ("code_ce", "the next-code CE", "Code CE", 61),
                ("kl", "the posterior KL", "KL", 62),
                ("recon", "the VAE reconstruction", "Recon", 63),
            )
        },
        "calm_recon_rel": {
            "description": (
                "Continuous codec's relative squared reconstruction error on "
                "the patch features. 1.0 is predict-zero. This is the "
                "objective the bare linear posterior never had."
            ),
            "chart": {
                "title": "CALM Recon Error",
                "y_label": "relative squared error",
                "group": "calm_arm",
                "order": 15,
            },
        },
        "calm_energy_cond_gap": {
            "description": (
                "Energy score on misaligned targets minus the aligned one. "
                "Near 0 means the head ignores its conditioning and models "
                "the marginal."
            ),
            "chart": {
                "title": "CALM Conditioning Gap",
                "y_label": "mismatched - matched energy",
                "group": "calm_arm",
                "order": 25,
            },
        },
        "calm_arm_gate": {
            "description": (
                "Learned share of ARM_CEILING the continuous arm may occupy. "
                "Riding to the ceiling means the cap is binding and the arm "
                "wants more room than a residual is allowed to take."
            ),
            "chart": {
                "title": "CALM Arm Gate",
                "y_label": "ceiling fraction",
                "group": "calm_arm",
                "order": 35,
            },
        },
        "calm_vote_lift": {
            "description": (
                "Distance from the voted latent to the MEAN of all "
                "proposals, in units of their spread. Near 0 means the vote is "
                "decorative and this became a conditional-mean estimator."
            ),
            "chart": {
                "title": "CALM Vote Lift",
                "y_label": "|voted - mean| / spread",
                "group": "calm_arm",
                "order": 50,
            },
        },
        "calm_vote_margin": {
            "description": (
                "Modal code's share of the draws at generation. The count IS "
                "the temperature (CALM uses n = round(1/T)). Near 1/K is a "
                "vote over noise; a plurality means it is selecting."
            ),
            "chart": {
                "title": "CALM Vote Margin",
                "y_label": "modal share of draws",
                "group": "calm_arm",
                "order": 40,
            },
        },
    }
