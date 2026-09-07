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
from praxis.heads.energy import EnergyHead
from praxis.losses.energy_score import energy_score_loss

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
# training. CALM's defaults; both are Monte Carlo estimators, and more draws
# reduce variance on the estimate rather than changing what is estimated.
ENERGY_SAMPLES_N: int = 8
ENERGY_SAMPLES_M: int = 16

# Free-bits floor for the continuous arm's KL, in nats per latent dimension.
# Below this the KL is not penalized, which stops the posterior collapsing onto
# the prior in the early steps when the discrete arm alone can explain the data
# - the failure this arm exists to avoid.
FREE_BITS: float = 0.05

# Weight on the dense code cross-entropy. Comparable to the energy term so it
# dominates the early steering (the score alone never concentrated the head),
# and it is the same scale as the MSE anchor it replaces. Fixed, model-agnostic.
CODE_CE_WEIGHT: float = 5.0

# Weight on the continuous arm's KL. Small: the reconstruction pressure comes
# from the byte decoder's CE, and an over-weighted KL simply mutes the arm.
KL_WEIGHT: float = 1e-3


class AbstractinatorCALM(AbstractinatorEncoder):
    """Abstractinator with a continuous CALM arm and a count-based vote.

    Everything the parent does is unchanged; this adds a Gaussian posterior
    beside the quantizer, an energy head over the trunk output, and the dense
    code objective that pays for it.
    """

    def __init__(
        self,
        config: ConfigType,
        *,
        energy_hidden_ratio: float = 1.0,
        energy_blocks: int = 3,
        vote_samples: int = VOTE_SAMPLES,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        D = config.hidden_size
        self.vote_samples = int(vote_samples)

        # The continuous arm: a Gaussian posterior over the SAME patch features
        # the quantizer sees. Deliberately not CALM's token-chunk VAE - that
        # compresses K token embeddings and owns its own table, which is a job
        # the Abstractinator's local encoder already did. What is needed here is
        # only the continuous channel, so this is the posterior and nothing else.
        # The byte decoder supplies the reconstruction pressure, so no second
        # decoder exists to disagree with the one already there.
        self.posterior = nn.Linear(D, 2 * D)
        # Start the arm SILENT so the encoder begins bit-identical to its parent
        # and the run is a clean A/B rather than a reroll. Zeroing the whole
        # layer is not enough and is the trap: it zeroes mu AND logvar, and
        # logvar 0 means sigma = 1, so z_c would be a standard normal added to
        # every patch latent - measured at ||z_c||/||z_q|| = 1.24 at step 0.
        # The log-variance bias goes to the clamp floor instead, so sigma is
        # ~3e-4 and z_c really is ~0.
        nn.init.zeros_(self.posterior.weight)
        nn.init.zeros_(self.posterior.bias)
        with torch.no_grad():
            self.posterior.bias[D:].fill_(-8.0)

        hidden = max(1, int(D * energy_hidden_ratio))
        self.energy_head = EnergyHead(
            cond_dim=D,
            noise_dim=D,
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

        self._pending: Dict[str, torch.Tensor] = {}
        self._calm_diag: Dict[str, float] = {}
        self._last_latent: Optional[torch.Tensor] = None
        self._last_posterior: Optional[tuple] = None
        self._last_stage_indices: Optional[list] = None

    # ── training ───────────────────────────────────────────────────────────

    def _post_downsample(self, h, aux_loss):
        """Parent's RVQ, plus the continuous arm added residually."""
        z_q, aux = super()._post_downsample(h, aux_loss)

        stats = self.posterior(h)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if self.training:
            z_c = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z_c = mu

        # Free-bits KL: per-dimension, floored, then summed. A plain KL drives
        # this arm to exactly zero early on, when the discrete arm alone already
        # explains the data - which would silently delete the thing under test.
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl = kl.clamp_min(FREE_BITS).sum(-1).mean()
        self._last_posterior = (mu, logvar)
        self._pending["calm_kl"] = KL_WEIGHT * kl

        z = z_q + z_c
        self._last_latent = z
        self._calm_diag["calm_arm_ratio"] = float(
            z_c.detach().norm() / z_q.detach().norm().clamp_min(1e-6)
        )
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
                acc += float(
                    (logits.detach().argmax(-1) == target).float().mean()
                )
            n = max(1, min(len(digits), len(self.code_heads)))
            self._pending["calm_code_ce"] = CODE_CE_WEIGHT * (ce / n)
            self._calm_diag["calm_code_acc"] = acc / n

        # ── the continuous arm: CALM's energy score ────────────────────────
        z = self._last_latent
        if z is None or self._last_posterior is None:
            return
        target = z[:, 1:, :].detach()
        B, P, D = target.shape
        # Model draws: N independent noise vectors per position.
        noise = torch.randn(B, P, ENERGY_SAMPLES_N, D, device=h.device, dtype=h.dtype)
        cond = h_cond.unsqueeze(2).expand(B, P, ENERGY_SAMPLES_N, D)
        proposals = self.energy_head(
            cond.reshape(-1, D), noise.reshape(-1, D)
        ).view(B, P, ENERGY_SAMPLES_N, D)
        # Target set: M draws from the NEXT patch's own posterior, which this
        # encoder has because the continuous arm is a distribution and not a
        # point. That keeps the energy score a distributional match (CALM's
        # construction) rather than a regression onto one vector - which would
        # be mean-seeking, the defect this design exists to avoid. Detached, so
        # the score trains the head and never the posterior.
        mu, logvar = self._last_posterior
        mu_n = mu[:, 1:, :].detach().unsqueeze(2)
        std_n = (0.5 * logvar[:, 1:, :]).exp().detach().unsqueeze(2)
        targets = mu_n + torch.randn(
            B, P, ENERGY_SAMPLES_M, D, device=h.device, dtype=mu_n.dtype
        ) * std_n
        # The quantized half carries no posterior, so it enters as the shared
        # offset it is: the arm's draws orbit the discrete code.
        targets = targets + (target - mu[:, 1:, :].detach()).unsqueeze(2)
        self._pending["calm_energy"] = energy_score_loss(proposals, targets)

        with torch.no_grad():
            # Does the head CONCENTRATE? Cross-sample agreement in code space:
            # quantize every proposal and count how often two draws land on the
            # same code. ~0 was the marginal-collapse signature.
            codes = self._quantize_to_codes(proposals.reshape(-1, D)).view(
                B, P, ENERGY_SAMPLES_N
            )
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
        B, D = h_cond.shape
        n_draws = self.vote_samples
        cond = h_cond.unsqueeze(1).expand(B, n_draws, D).reshape(-1, D)
        noise = torch.randn(B * n_draws, D, device=h_cond.device, dtype=h_cond.dtype)
        proposals = self.energy_head(cond, noise).view(B, n_draws, D)
        codes = self._quantize_to_codes(proposals.reshape(-1, D)).view(B, n_draws)

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
            out[b] = members[torch.randint(len(members), (1,), device=members.device)][0]
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
            logw = (
                torch.lgamma(c + 1)
                - math.lgamma(n + 1)
                - torch.lgamma(c - n + 1)
            )
            probs = torch.softmax(logw - logw.max(), dim=0)
            return vals[keep][int(torch.multinomial(probs, 1).item())]
        return vals[int(counts.argmax())]

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
                "||z_c|| / ||z_q||: the continuous arm's contribution beside "
                "the quantized one. Starts ~0.02 (near-silent, so this is an "
                "A/B); staying there means it never earned its parameters."
            ),
            "chart": {
                "title": "CALM Arm Ratio",
                "y_label": "||z_c|| / ||z_q||",
                "group": "calm_arm",
                "order": 30,
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
