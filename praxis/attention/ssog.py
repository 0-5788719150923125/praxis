"""SSOG attention: a query-steered Gaussian field over relative lag, no Q, no K.

Port of Sum of Separable Gaussians (Pisoni, https://github.com/4rtemi5/ssog,
vision only) to a causal 1D sequence. Content never scores content. Each head
owns ``NUM_ATOMS`` Gaussian atoms over the lag ``d = q_idx - kv_idx`` - three
numbers each: centre ``mu`` (how far back to look), width ``sigma`` and mixture
weight ``lambda`` - and the attention logit from a query to a key is the log of
that mixture at their lag, sharpened by one learned temperature:

    logit(q, k) = logsumexp_r( log lambda_r + log N(q - k; mu_r, sigma_r) ) / tau

softmaxed over the causal keys and applied to V. Only V is projected; the QK
projections and their d^2 parameters are gone.

Steering ("lookat"): a zero-initialised probe on the QUERY token predicts bounded
residuals on mu, sigma and lambda behind cold softplus(-8) gates, so the field
starts frozen and the model opens the content taps itself. The residual on mu
lives in softplus's raw space so a steered atom can never point into the
future - under a causal mask an atom with ``mu < 0`` collapses onto lag 0 and
its gradient dies, so non-negativity is enforced rather than hoped for.

Differences from the reference on purpose:

* 1D and causal. The 2D separability trick has nothing to factorise here, so
  the cost is that of ordinary attention: FlexAttention with a ``score_mod``
  that evaluates the mixture from per-query captured tensors (each atom its
  own tensor - indexing ``mu[b, h, q, r]`` with a python int does not trace),
  a materialised ``[B, H, T, T]`` path on CPU.
* No ghost token. Softmax1's always-visible zero-logit ghost would take
  roughly half of a Gaussian field's mass; a field wants an explicit learned
  null atom for that, which is deliberately left out of this first version.
* Mixture-then-softmax (the reference's stated maths) rather than the code's
  per-atom softmax then lambda-mix; the reference reports the two within noise
  and the former is one kernel call.

Positional encoding is irrelevant (there is no Q/K to rotate); ``pos_type``
reports ``"ssog"``. ``num_queries`` is corrected to 1 by ``patch_config`` since
no query heads are built. See ``next/`` for the language-modeling expectation:
position-addressed, never content-addressed - a hybrid arm, not a replacement.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_ATOMS: int = 4  # Gaussian atoms per head (the reference's sweet spot)
SIGMA_FLOOR: float = 0.25  # minimum atom width, in tokens
MAX_OFFSET: float = 8.0  # bound on per-query mu travel, in tokens (raw space)
COLD_GATE_INIT: float = -8.0  # softplus(-8) ~ 3e-4: steering starts closed
MU_INIT_MIN: float = 0.5  # nearest atom's lag at init, in tokens
MU_INIT_MAX: float = 32.0  # farthest atom's lag at init, in tokens
SIGMA_Q: float = 0.35  # atom width as a fraction of its own lag (constant-Q)
SIGMA_MIN_EXCESS: float = 0.05  # smallest width the ladder asks for, above floor
TEMPERATURE_RAW_INIT: float = -1.0  # softplus(-1) + 0.5 ~ 0.81, slightly sharp
QK_DUMMY_DIM: int = 16  # flex needs Q/K tensors; theirs are zeros
NULL_EMA_DECAY: float = 0.99  # smoothing on both realized null buffers
# Null-atom logit at init, calibrated against the MEASURED denominator rather
# than guessed. The field's logsumexp sits near -0.83 to -0.97 across sequence
# lengths 64-512, so the null's share at init is sigmoid-close to
# exp(l_0 - lse): l_0 = -2 gives ~25%, -3 ~11%, -4 ~4.5%, -5 ~1.7%.
#
# -4.0 is the choice: ~5% is near-identity, which is this lineage's discipline
# (the arm starts inert and the model opens it), while staying far from
# saturation. Going strongly negative would be the cold-gate trap a second
# time - at l_0 = -20, sigmoid(lse - l_0) pins at 1.0 and the gradient dies,
# which is exactly how the reference's softplus(-8) steering gate spent 11.7k
# steps welded shut.
NULL_LOGIT_INIT: float = -4.0
GEOM_BINS: int = 192  # lag samples per geometry row (a live snapshot, so free)
GEOM_LAG_MIN: float = 0.5  # nearest lag plotted, in tokens
GEOM_LAG_MAX: float = 512.0  # farthest lag plotted, in tokens
GEOM_MAX_HOPS: int = 12  # cap on cascade rows, whatever config.depth says
_EPS: float = 1e-4
_LOG_2PI: float = math.log(2.0 * math.pi)


def _inv_softplus(y: Tensor) -> Tensor:
    """Inverse of softplus, without overflowing on large lags.

    The direct form ``log(expm1(y))`` overflows float32 at y > ~88, which is
    reachable the moment a ladder is initialised past that many tokens (it
    silently produced ``inf`` centres). ``log(e^y - 1) = y + log(1 - e^-y)`` is
    the same function, stable for every positive y.
    """
    return y + torch.log(-torch.expm1(-y))


def _log_kernel(d: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """log N(d; mu, sigma^2)."""
    return -0.5 * _LOG_2PI - torch.log(sigma) - (d - mu) ** 2 / (2.0 * sigma**2)


class SSOGAttention(nn.Module):
    """Sum-of-Gaussians attention field over causal lag (see module docstring).

    This class is the FAITHFUL port and is meant to stay that way; variants
    subclass it. The three knobs below are class attributes rather than direct
    reads of the module constants so a subclass can reshape the field without
    reimplementing ``__init__``, and ``_build_field`` / ``_field`` are the two
    seams a variant overrides. See ``praxis/attention/arc_ssog.py``.
    """

    default_atoms: int = NUM_ATOMS
    gate_init: float = COLD_GATE_INIT
    mu_init_max: float = MU_INIT_MAX

    # Dashboard declaration. The Dynamics tab builds its cards from this map,
    # so a key absent here logs a column nobody ever sees - which is exactly
    # what happened when these lived in COMPOSITE_METRIC_REGISTRY, whose cards
    # read metrics.db while training_metrics() writes dynamics.db.
    metric_descriptions = {
        **{
            f"ssog_mu_a{r}": {
                "description": (
                    "Centre lag of Gaussian atom "
                    f"{r}, in tokens: how far back it looks. Atoms start on a "
                    "geometric ladder (0.5 .. 32) and can only travel a few "
                    "sigma from there, because the gradient on mu arrives "
                    "weighted by the attention mass the atom already puts at "
                    "that lag. So these lines show how far the field MOVED, "
                    "not how far it could have."
                ),
                "chart": {
                    "title": "SSOG Atom Lags",
                    "y_label": "Lag (tokens)",
                    "y_scale": "logarithmic",
                    "group": "ssog_field",
                    # Places the whole section where the other attention
                    # diagnostics live (arc, infini all declare 30).
                    "group_order": 30,
                    "order": 10 + r,
                    "series_group": "ssog_mu",
                    "series_label": f"atom {r}",
                },
            }
            for r in range(NUM_ATOMS)
        },
        **{
            f"ssog_sigma_a{r}": {
                "description": (
                    f"Width of Gaussian atom {r}, in tokens. At the 0.25 floor "
                    "with its lag near 1 the atom is a previous-token head; "
                    "growing without bound it is a causal bag of words. Both "
                    "are legitimate, and a MIX across atoms is the picture the "
                    "reference reports."
                ),
                "chart": {
                    "title": "SSOG Atom Widths",
                    "y_label": "Sigma (tokens)",
                    "y_scale": "logarithmic",
                    "group": "ssog_field",
                    "order": 20 + r,
                    "series_group": "ssog_sigma",
                    "series_label": f"atom {r}",
                },
            }
            for r in range(NUM_ATOMS)
        },
        **{
            f"ssog_lambda_a{r}": {
                "description": (
                    f"Mixture weight of atom {r}, before per-query steering. "
                    "Uniform means the field declined to prefer a scale; one "
                    "atom near 1.0 means the head collapsed onto a single lag "
                    "and the rest of the ladder is dead weight. The far atom "
                    "keeping its weight is the only evidence this block is "
                    "doing anything at range."
                ),
                "chart": {
                    "title": "SSOG Atom Mixture Weights",
                    "y_label": "lambda",
                    "group": "ssog_field",
                    "order": 30 + r,
                    "series_group": "ssog_lambda",
                    "series_label": f"atom {r}",
                },
            }
            for r in range(NUM_ATOMS)
        },
        **{
            f"ssog_gate_{name}": {
                "description": (
                    f"How far content is allowed to move the atoms' {name}. "
                    "All three gates start cold at ~3e-4, so the field begins "
                    "as a fixed causal convolution and the model decides "
                    "whether to open them at all. Flat at the init value for a "
                    "whole run is a FINDING, not a failure: it says a purely "
                    "positional field was all this stack asked for. The "
                    "reference saw lambda open hardest, and more so with depth."
                ),
                "chart": {
                    "title": "SSOG Steering Gates",
                    "y_label": "softplus(raw gate)",
                    "y_scale": "logarithmic",
                    "group": "ssog_field",
                    "order": 40 + i,
                    "series_group": "ssog_gates",
                    "series_label": name,
                },
            }
            for i, name in enumerate(("mu", "sigma", "lambda"))
        },
        "ssog_temperature": {
            "description": (
                "The one learned sharpening knob, floored at 0.5. Tempering a "
                "Gaussian keeps it Gaussian with variance scaled by tau, so "
                "this is a width multiplier on the whole field at once. "
                "Driving hard into the floor is the cue that the head is "
                "trying to manufacture an outlet it does not have: SSOG has no "
                "null atom, so a query wanting to contribute nothing can only "
                "approximate it by going needle-narrow onto one key."
            ),
            "chart": {
                "title": "SSOG Field Temperature",
                "y_label": "tau",
                "group": "ssog_field",
                "order": 50,
            },
        },
        "ssog_geometry": {
            "description": (
                "The reference's own claim is that a head stops being a "
                "heatmap and a shrug: it is a few blobs you can plot and read "
                "with a ruler. This is that plot, 1D and causal. Each band is "
                "one Gaussian atom with its mixture weight folded in (so an "
                "atom the field stopped using goes dark rather than staying a "
                "bright blob nobody reads), nearest at the bottom, the summed "
                "mixture on top; x is lag on a geometric axis from 0.5 to 512 "
                "tokens. Separated blobs is the ladder as initialized; "
                "everything crowding the left edge is a previous-token head; "
                "one band smeared across the width is a causal bag of words. "
                "The axis deliberately runs past the sequence length, so mass "
                "the field has walked outside its own window shows as such. "
                "The field's SHAPE, from the learned atoms - not a measured "
                "attention row, so the causal truncation and the per-query "
                "steering are not in it."
            ),
            "snapshot": {
                "title": "Attention Geometry",
                "renderer": "heatmap_2d",
                "color_scale": "linear",
                "group": "ssog_field",
                "order": 100,
            },
        },
        "ssog_cascade": {
            "description": (
                "The reference plots geometry per LAYER, and that is the one "
                "axis which does not port: this field is depth-SHARED, so "
                "there is no per-layer field to draw. The analogue for a "
                "recurrent-depth model is the cascade. An attention row here "
                "IS the mixture density, Gaussians are closed under "
                "convolution, and residual connections keep every hop count "
                "live at once - so one shared field gives a scale-space "
                "pyramid whose h-th level is the h-fold self-convolution, mean "
                "lag h*mu and width sqrt(h)*sigma. Bottom band is one hop, top "
                "is the full depth; each is normalized to its own peak, since "
                "the question is where a hop lands, not how thin it is. Read "
                "it as reach: the top band is the farthest this block can see, "
                "and if it sits inside a handful of tokens then nothing here "
                "is doing long range. Ignores the causal window and whatever "
                "the residual stream does between passes."
            ),
            "snapshot": {
                "title": "Field by Recurrent Depth",
                "renderer": "heatmap_2d",
                "color_scale": "linear",
                "group": "ssog_field",
                "order": 101,
            },
        },
    }

    def __init__(
        self,
        config,
        num_atoms: Optional[int] = None,
        mu_init_max: Optional[float] = None,
        null_atom: bool = False,
    ) -> None:
        super().__init__()
        self.patch_config(config)
        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = (
            getattr(config, "head_size", None) or hidden_size // self.num_heads
        )
        self.num_atoms = int(num_atoms or self.default_atoms)
        # Instance override so a registry PROFILE can set the ladder's span
        # alongside its atom count; the two are one decision, not two.
        self.mu_init_max = float(mu_init_max or type(self).mu_init_max)
        self.null_atom = bool(null_atom)
        self.causal = config.causal
        self.window_size = getattr(config, "window_size", None)
        self.dropout_p = config.dropout
        self.pos_type = "ssog"

        H, R = self.num_heads, self.num_atoms
        self.value = nn.Linear(hidden_size, H * self.head_dim, bias=False)
        self.output = nn.Linear(H * self.head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(self.dropout_p)

        self._build_field(hidden_size, H, R)

        # Display grid for the geometry cards: one FIXED geometric ladder of
        # lags, shared by both, so a step slider compares like with like across
        # training. It deliberately runs past the sequence length - a field that
        # has walked its mass beyond the window is spending it on nothing, and
        # that is only visible if the axis extends far enough to show it.
        self.cascade_hops = max(1, min(int(getattr(config, "depth", 1)), GEOM_MAX_HOPS))
        ramp = torch.linspace(0.0, 1.0, GEOM_BINS)
        self.register_buffer(
            "geom_lags",
            GEOM_LAG_MIN * (GEOM_LAG_MAX / GEOM_LAG_MIN) ** ramp,
            persistent=False,
        )

        # Snapshot payload, published by the training thread (see
        # dashboard_snapshots). Seeded here so a card exists before the first
        # metrics tick; at construction the module is still on the host, so
        # this costs nothing and blocks on nothing.
        self._snapshot_cache: dict = self._build_snapshots()

        self.flex_attention = None
        self.create_block_mask = None
        self.and_masks = None
        self._aux_request = None
        # `--no-compile` means no compile, including ours. The flex path below
        # compiles ITSELF whatever the rest of the model does, because eager
        # flex_attention cannot backprop through captured score_mod tensors and
        # V together (a vmap limitation in its dense backward) and the steering
        # probes ARE captured tensors that need gradient. That self-compile is
        # invisible under an outer torch.compile, which dynamo inlines - but
        # under `no_compile: true` it is the ONLY thing compiling, and its cost
        # scales with the atom count: one captured tensor per atom per steered
        # family, so a 12-atom field traces 36 of them and inductor sits on the
        # first forward for a very long time. The materialised path needs no
        # compile, differentiates in plain eager, and is affordable here because
        # the batch schedule holds B*T^2 constant across curriculum tiers
        # (micro_rows = base_rows // m^2 against T = block_size * m), which puts
        # the logits tensor at ~1 MB at every tier. So honour the flag.
        self.compile_flex = not bool(getattr(config, "no_compile", False))
        if self.compile_flex:
            try:
                from torch.nn.attention.flex_attention import (
                    and_masks,
                    create_block_mask,
                    flex_attention,
                )

                self.flex_attention = torch.compile(flex_attention)
                self.create_block_mask = create_block_mask
                self.and_masks = and_masks
                try:
                    from torch.nn.attention.flex_attention import AuxRequest

                    self._aux_request = AuxRequest(lse=True)
                except ImportError:  # torch < 2.10 keeps return_lse
                    self._aux_request = None
            except ImportError:
                pass
        self.block_mask_cache = {}

    @classmethod
    def patch_config(cls, config) -> None:
        """No query heads are built, so ``num_queries`` must read 1. Idempotent."""
        if getattr(config, "num_queries", 1) != 1:
            config.num_queries = 1

    def _build_field(self, hidden_size: int, H: int, R: int) -> None:
        """Create the field and its steering probe.

        The seam a variant overrides to reshape the field (e.g. give it a
        depth axis). Reads ``self.gate_init`` / ``self.mu_init_max`` rather
        than the module constants so a subclass changes them by class
        attribute. Called once, from ``__init__``.
        """
        # The field. Atoms start on a GEOMETRIC ladder of lags (0.5 .. 32 for
        # four atoms), jittered multiplicatively, so the heads break symmetry
        # from step 0 instead of all staring at the same lag; mu is
        # softplus-parametrised so it is never negative.
        #
        # The ladder is geometric rather than linear because MU CANNOT TRAVEL
        # FAR BY GRADIENT DESCENT. d/dmu of the log kernel is (d - mu) / sigma^2,
        # linear in the residual, but it arrives weighted by the attention mass
        # this atom puts at lag d, which is Gaussian-small a few sigma out. So an
        # atom initialised at lag 3.5 will never discover lag 200: the ladder set
        # here IS the run's reachable receptive field, fixed for its lifetime.
        # A linear stagger (0.5, 1.5, 2.5, 3.5) caps the whole field at ~4 tokens
        # direct, ~21 through a depth-6 recurrent cascade, and any long-range
        # result then measures the init rather than the mechanism.
        #
        # Widths follow the lags (constant-Q, sigma ~ SIGMA_Q * mu) instead of a
        # flat 0.72. A far atom needs a basin, not a needle: a needle at lag 32
        # collects almost no softmax mass, so it gets almost no gradient either,
        # which would defeat the ladder before the first step. The near atoms sit
        # on the floor and stay previous-token sharp.
        ramp = torch.linspace(0.0, 1.0, R) if R > 1 else torch.zeros(1)
        init_mu = MU_INIT_MIN * (self.mu_init_max / MU_INIT_MIN) ** ramp  # (R,)
        init_mu = init_mu.repeat(H, 1) * torch.exp(0.1 * torch.randn(H, R))
        self.raw_mu = nn.Parameter(_inv_softplus(init_mu.clamp_min(0.05)))
        init_sigma = (SIGMA_Q * init_mu - SIGMA_FLOOR).clamp_min(SIGMA_MIN_EXCESS)
        self.raw_sigma = nn.Parameter(_inv_softplus(init_sigma))
        self.log_lambda = nn.Parameter(torch.zeros(H, R))
        # Shape [1], never 0-dim: schedule-free optimizers swap parameters by
        # `x.view(torch.uint8).bitwise_xor_(...)`, and a 0-dim tensor cannot be
        # viewed as a narrower dtype ("self.dim() cannot be 0 to view Float as
        # Byte"). Broadcasting is unaffected.
        self.raw_temperature = nn.Parameter(torch.tensor([TEMPERATURE_RAW_INIT]))

        # Steering: one zero-init probe per token -> (mu, sigma, lambda) residuals
        # for every atom of every head, each family behind its own cold gate.
        self.steer = nn.Linear(hidden_size, H * R * 3, bias=True)
        nn.init.zeros_(self.steer.weight)
        nn.init.zeros_(self.steer.bias)
        self.raw_gate = nn.Parameter(torch.full((3,), self.gate_init))
        self._build_null(H)

    def _build_null(self, H: int) -> None:
        """The learned null logit, one per head. See ``_apply_null``."""
        self.raw_null = (
            nn.Parameter(torch.full((H,), NULL_LOGIT_INIT)) if self.null_atom else None
        )
        self._register_null_buffers()

    def _register_null_buffers(self) -> None:
        """Both realized-value buffers, registered in ONE place.

        Every number on the null card has to be read from the FORWARD, so both
        buffers live here rather than beside whichever ``_build_null`` a
        subclass happens to write. ``arc_ssog`` keeping its own copy of one of
        them is how ``null_logit`` came to report a different tensor than
        ``null_share``: see ``_note_null_logit``.
        """
        slots = self._null_slots()
        # Realized share, EMA'd in the forward. The PARAMETER says what the head
        # asked for; this says what the softmax actually gave it, which is the
        # position-dependent part and the only one worth reading.
        self.register_buffer("null_share", torch.zeros(slots), persistent=False)
        # Realized LOGIT, EMA'd the same way, and NOT a view of ``raw_null``.
        self.register_buffer(
            "null_logit_seen",
            torch.full((slots,), float(NULL_LOGIT_INIT)),
            persistent=False,
        )

    def _null_slots(self) -> int:
        """How many null-share slots to track. One per depth where the field is
        per-depth, one overall where it is shared."""
        return 1

    # ------------------------------------------------------------------ field
    def _field(
        self, x: Tensor, current_depth: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-query atom parameters, all ``[B, H, T, R]`` float32, plus tau.

        ``current_depth`` is unused here - the faithful field is shared across
        every recurrent pass - and is in the signature for variants that are
        not (see ``ArcSSOGAttention``).
        """
        B, T, _ = x.shape
        H, R = self.num_heads, self.num_atoms
        steer = self.steer(x).float().view(B, T, H, R, 3).permute(0, 2, 1, 3, 4)
        gate = F.softplus(self.raw_gate.float())  # (3,)
        raw_mu = self.raw_mu.float()[None, :, None, :]
        mu = F.softplus(raw_mu + gate[0] * MAX_OFFSET * torch.tanh(steer[..., 0]))
        sigma0 = F.softplus(self.raw_sigma.float()) + _EPS + SIGMA_FLOOR
        sigma = sigma0[None, :, None, :] * torch.exp(
            gate[1] * torch.tanh(steer[..., 1])
        )
        loglam = torch.log_softmax(
            self.log_lambda.float()[None, :, None, :]
            + gate[2] * torch.tanh(steer[..., 2]),
            dim=-1,
        )
        tau = F.softplus(self.raw_temperature.float()) + 0.5
        return mu, sigma, loglam, tau

    # -------------------------------------------------------------- flex path
    def _mask_mod(self):
        window = self.window_size

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        if window is None:
            return causal

        def within(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= window

        return self.and_masks(causal, within)

    def _block_mask(self, T: int, device: torch.device):
        key = (T, str(device))
        if key not in self.block_mask_cache:
            self.block_mask_cache[key] = self.create_block_mask(
                self._mask_mod(), B=None, H=None, Q_LEN=T, KV_LEN=T, device=device
            )
        return self.block_mask_cache[key]

    def _flex(self, v: Tensor, mu, sigma, loglam, tau) -> Tensor:
        B, H, T, _ = v.shape
        # One captured tensor per atom: the closure indexes them by (b, h, q_idx).
        mus = [mu[..., r].contiguous() for r in range(self.num_atoms)]
        sigs = [sigma[..., r].contiguous() for r in range(self.num_atoms)]
        lams = [loglam[..., r].contiguous() for r in range(self.num_atoms)]
        # Captured as a full [B, H, T] tensor, not a 0-dim scalar: inductor's
        # flex backward cannot allocate a grad buffer for a scalar capture.
        inv_tau = (1.0 / tau).expand(B, H, T).contiguous()

        def score_mod(score, b, h, q_idx, kv_idx):
            d = (q_idx - kv_idx).to(torch.float32)
            acc = None
            for m, s, l in zip(mus, sigs, lams):
                term = l[b, h, q_idx] + _log_kernel(d, m[b, h, q_idx], s[b, h, q_idx])
                acc = term if acc is None else torch.logaddexp(acc, term)
            return score + acc * inv_tau[b, h, q_idx]

        q_dummy = v.new_zeros(B, H, T, QK_DUMMY_DIM)
        k_dummy = v.new_zeros(B, H, T, QK_DUMMY_DIM)
        block_mask = self._block_mask(T, v.device) if self.causal else None
        # The null scale needs the attention denominator, and flex already has
        # it; asking costs nothing when the null is off. `return_lse` is
        # deprecated from torch 2.10 in favour of `return_aux`, so prefer the
        # new spelling where it exists and keep the old one working.
        kwargs = dict(score_mod=score_mod, block_mask=block_mask)
        if self._aux_request is not None:
            out, aux = self.flex_attention(
                q_dummy, k_dummy, v, **kwargs, return_aux=self._aux_request
            )
            return out, aux.lse
        return self.flex_attention(q_dummy, k_dummy, v, **kwargs, return_lse=True)

    # ------------------------------------------------------- materialised path
    def _materialised(self, v: Tensor, mu, sigma, loglam, tau) -> Tensor:
        B, H, T, _ = v.shape
        pos = torch.arange(T, device=v.device, dtype=torch.float32)
        d = (pos[:, None] - pos[None, :])[None, None]  # [1, 1, T, T] = q - k
        logits = None
        for r in range(self.num_atoms):
            term = loglam[..., r, None] + _log_kernel(
                d, mu[..., r, None], sigma[..., r, None]
            )  # [B, H, T, T]
            logits = term if logits is None else torch.logaddexp(logits, term)
        logits = logits / tau
        if self.causal:
            allowed = d >= 0
            if self.window_size is not None:
                allowed = allowed & (d <= self.window_size)
            logits = logits.masked_fill(~allowed, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)  # [B, H, T]
        weights = torch.softmax(logits, dim=-1).to(v.dtype)
        return weights @ v, lse

    def _apply_null(self, out: Tensor, lse: Tensor, current_depth: int) -> Tensor:
        """Scale the output by the share the real keys keep against the null.

        The null is an extra softmax column whose VALUE is zero, so it steals
        probability mass and contributes nothing. Writing its share out,

            null   = exp(l_0) / (exp(l_0) + Z),      Z = sum_k exp(logit_k)
            keep   = Z / (Z + exp(l_0)) = sigmoid(logsumexp(logits) - l_0)

        so the whole thing is one sigmoid on the attention denominator, which
        both paths already compute. No extra column, no kernel change.

        WHY IT IS WORTH HAVING HERE, and it is not the dilution argument: this
        stack has NO absolute position anywhere below the head - the logit is a
        function of lag alone - so a query cannot represent "I am near the start
        and there is nothing that far back". Without a null, an atom centred at
        lag 32 queried at position 5 has its truncated tail renormalised onto
        the oldest token, which is a sink indistinguishable from a real
        long-range read. With one, Z is tiny exactly there (a far atom puts
        almost no density on the few available lags), so a CONSTANT l_0 absorbs
        most of the mass; deep in the sequence the same atom sits in-window, Z
        is large, and the null gets almost nothing. One learned scalar buys
        position-dependent abstention without a position input.

        What it does NOT do is restore concentration between atoms: ``keep`` is
        a common factor, so every real atom keeps its 1/R share of what is left.
        """
        if self.raw_null is None:
            return out
        null = self._null_logit(current_depth)  # [H]
        keep = torch.sigmoid(lse.float() - null[None, :, None])  # [B, H, T]
        self._note_null_share(1.0 - keep, current_depth)
        self._note_null_logit(null, current_depth)
        return out * keep.unsqueeze(-1).to(out.dtype)

    def _null_logit(self, current_depth: int) -> Tensor:
        """``[H]`` null logit for this pass. Depth-shared in the faithful port."""
        return self.raw_null.float()

    def _note_null_share(self, share: Tensor, current_depth: int) -> None:
        """EMA of the mass actually going nowhere, per depth slot.

        The PARAMETER says what the head asked for; this says what the softmax
        gave it, which is the position-dependent half and the only one worth
        reading.
        """
        self._note_ema(self.null_share, share, current_depth)

    def _note_null_logit(self, logit: Tensor, current_depth: int) -> None:
        """EMA of the null logit AS APPLIED, which is not ``raw_null``.

        A modular-SMEAR router targets this parameter, so the tensor the
        sigmoid above actually sees is ``base + sum_e w_e * delta_e``,
        materialised by ``functional_call`` for the duration of the forward.
        ``training_metrics`` runs OUTSIDE that reparametrisation, so reading
        ``self.raw_null`` there returns the base alone - on the -s run arm 0
        carried +2.0 at depth 0, and the card under-reported by two logits
        while sitting next to a share measured from the merged value. Two
        halves of one card, disagreeing.

        Recording it here costs one buffer write and is the only place the
        merged tensor exists.
        """
        self._note_ema(self.null_logit_seen, logit, current_depth)

    def _note_ema(self, buffer: Tensor, value: Tensor, current_depth: int) -> None:
        """Shared decay for the realized-value buffers, so the halves of the
        null card are smoothed identically and stay comparable."""
        with torch.no_grad():
            v = value.mean().detach()
            if not torch.isfinite(v):
                return
            idx = min(int(current_depth), buffer.numel() - 1)
            buffer[idx].mul_(NULL_EMA_DECAY).add_(
                (1.0 - NULL_EMA_DECAY) * v.to(buffer.device)
            )

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        B, T, _ = inputs.shape
        v = self.value(inputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        mu, sigma, loglam, tau = self._field(inputs, current_depth)
        # INFERENCE NEVER TAKES THE COMPILED PATH. The flex branch compiles
        # itself (see __init__), and `DecodeBackend.eval_mode` documents why the
        # decode loop must not run through torch.compile at all: it changes
        # python-level guard inputs every token, so compiled frames blow the
        # dynamo recompile limit. Compiling inside the module smuggled a
        # compiled frame back into a loop built to avoid them. It is also
        # simply broken here - flex's inference kernel requires a power-of-two
        # head dim, and at head_size 37 the Triton compile fails outright
        # ("Shape element 2 must be a power of 2"), so every decode step paid a
        # full compile attempt to arrive at an exception. The materialised path
        # is exact, needs no compile, and at generation batch sizes is cheap.
        use_flex = (
            self.flex_attention is not None
            and inputs.device.type != "cpu"
            and torch.is_grad_enabled()
        )
        if use_flex:
            out, lse = self._flex(v, mu, sigma, loglam, tau)
        else:
            out, lse = self._materialised(v, mu, sigma, loglam, tau)
        out = self._apply_null(out, lse, current_depth)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.dropout(self.output(out)), past_key_values, 0.0

    # ---------------------------------------------------------------- metrics
    def training_metrics(self) -> dict:
        """How open the steering taps are, and where each ATOM sits.

        Per atom, not averaged over atoms. The whole picture worth reading here
        is the MIX - a previous-token needle sitting next to a wide distant
        basin is the paper's result, and it averages to a middle lag that no
        atom occupies. HEADS are averaged, since every head starts on the same
        ladder and so atom index r means the same thing across them.
        """
        gate = F.softplus(self.raw_gate.detach().float())
        mu = F.softplus(self.raw_mu.detach().float())  # (H, R)
        sigma = F.softplus(self.raw_sigma.detach().float()) + _EPS + SIGMA_FLOOR
        lam = torch.softmax(self.log_lambda.detach().float(), dim=-1)
        out = {
            "ssog_gate_mu": gate[0].item(),
            "ssog_gate_sigma": gate[1].item(),
            "ssog_gate_lambda": gate[2].item(),
            "ssog_temperature": (
                F.softplus(self.raw_temperature.detach().float()) + 0.5
            )
            .mean()
            .item(),
            **(
                {
                    "ssog_null_logit": self.null_logit_seen.item(),
                    "ssog_null_share": self.null_share.item(),
                }
                if self.raw_null is not None
                else {}
            ),
        }
        for r in range(self.num_atoms):
            out[f"ssog_mu_a{r}"] = mu[:, r].mean().item()
            out[f"ssog_sigma_a{r}"] = sigma[:, r].mean().item()
            out[f"ssog_lambda_a{r}"] = lam[:, r].mean().item()

        # Publish the geometry from HERE, on the training thread, where a
        # device copy is ordinary rather than a cross-thread hazard.
        self._snapshot_cache = self._build_snapshots()
        return out

    def dashboard_snapshots(self) -> dict:
        """Hand back the payload the TRAINING thread already built.

        This runs in the web layer's snapshot producer thread, and it must not
        touch the model. Reading a live parameter from there means a device
        call - ``.cpu()`` is a blocking device-to-host copy just as much as a
        softmax is - and that thread then contends with training for the CUDA
        context and the allocator. Moving the arithmetic to the host was not
        enough: the COPY was the blocking part, and the run wedged again in the
        same method one line further down.

        So nothing here reads a parameter at all. ``training_metrics`` runs on
        the training thread and stashes a finished payload of plain lists,
        which is the pattern ``_rlct_landscape`` and ``_compute_profile``
        already use for exactly this reason. Rebinding a name is atomic, so a
        reader sees either the previous payload or the next one and never a
        torn one - provided we always build a NEW dict rather than mutate.
        """
        return self._snapshot_cache or {}

    def _build_snapshots(self) -> dict:
        """The two geometry heatmaps, as live grids.

        These are pictures of the CURRENT field, which is what the reference's
        figure is too - so they belong on the snapshot path (fetched live, at
        whatever resolution reads well) rather than in the per-step log. Sending
        them as logged metrics costs one database column per cell and buys a
        history nobody reads a heatmap for.
        """
        dens, mix = self._geometry()
        cascade = self._cascade()
        lag_range = [float(GEOM_LAG_MIN), float(GEOM_LAG_MAX)]
        # Row 0 renders at the BOTTOM, so the nearest atom and the shallowest
        # hop sit at the bottom and the field reads upward, like the harmonic
        # spectrum card and the paper's figures.
        geom = torch.cat([dens, mix[None, :]], dim=0)
        return {
            "ssog_geometry": {
                "grid": geom.tolist(),
                "grid_rows": int(geom.shape[0]),
                "grid_cols": GEOM_BINS,
                "x_range": lag_range,
                "y_range": [0, int(geom.shape[0])],
                "max_count": float(geom.max().item()),
            },
            "ssog_cascade": {
                "grid": cascade.tolist(),
                "grid_rows": int(cascade.shape[0]),
                "grid_cols": GEOM_BINS,
                "x_range": lag_range,
                "y_range": [1, int(cascade.shape[0])],
                "max_count": float(cascade.max().item()),
            },
        }

    # -------------------------------------------------------------- geometry
    def _atoms(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Head-averaged ``(mu, sigma, lambda)``, each ``[R]``, detached, ON CPU.

        The copy to host is the whole point, and it is not an optimisation.
        Everything below feeds ``dashboard_snapshots``, which the web layer's
        snapshot producer calls FROM A BACKGROUND THREAD while training runs.
        Issuing GPU work there - even a softmax over 12 numbers - contends with
        the training stream, and any device sync in that thread can block on it
        indefinitely. That is not hypothetical: it wedged a run for six hours,
        every watchdog dump showing the producer parked on the CUDA softmax
        this method used to run (praxis/callbacks/lightning/stall_watchdog.py).
        The field is a few dozen floats, so one host copy costs nothing and the
        rest of the geometry math is plain CPU tensors.
        """
        raw_mu, raw_sigma = self.raw_mu.detach(), self.raw_sigma.detach()
        log_lambda = self.log_lambda.detach()
        mu = F.softplus(raw_mu.float().cpu())
        sigma = F.softplus(raw_sigma.float().cpu()) + _EPS + SIGMA_FLOOR
        lam = torch.softmax(log_lambda.float().cpu(), dim=-1)
        return mu, sigma, lam

    def _geometry(self) -> Tuple[Tensor, Tensor]:
        """The learned field sampled on the display grid.

        Returns per-atom densities ``[R, GEOM_BINS]`` and their sum ``[GEOM_BINS]``,
        both scaled so the mixture peaks at 1. Lambda is folded in on purpose:
        an atom the mixture has stopped weighting is not part of the geometry
        any more, and should read as dark rather than as a bright blob nobody
        uses. This is the field's shape, not a measured attention row - the
        causal truncation and renormalization that the real softmax applies at
        each query are not here, and neither is per-query steering.
        """
        mu, sigma, lam = self._atoms()  # each [H, R]
        d = self.geom_lags.float().cpu()  # [G]
        dens = lam[..., None] * torch.exp(
            _log_kernel(d, mu[..., None], sigma[..., None])
        )  # [H, R, G]
        dens = dens.mean(dim=0)  # [R, G] - heads share a ladder, so r is comparable
        mix = dens.sum(dim=0)  # [G]
        scale = mix.max().clamp_min(_EPS)
        return dens / scale, mix / scale

    def _cascade(self) -> Tensor:
        """Where a ``h``-hop path through the field lands, for h = 1..depth.

        The field is depth-SHARED, so this stack has no per-layer geometry to
        plot the way the reference's vision model does - its layer axis is the
        one thing that does not port. The honest analogue for a recurrent-depth
        model is the CASCADE: the attention row is the (tempered, truncated)
        mixture density itself, Gaussians are closed under convolution, and
        residual connections mean every hop count 1..depth is live at once. So
        one shared field gives a scale-space pyramid whose h-th level is the
        h-fold self-convolution of the kernel - mean lag ``h * mu``, width
        ``sqrt(h) * sigma``. That is the "early layers are convolutions, late
        layers go global" progression, derived rather than learned.

        Computed on the integer lag lattice by repeated multiplication in the
        Fourier domain, on a window sized to the atoms' own reach so a far
        field cannot wrap around onto short lags. Each row is normalized to its
        own peak: the question a row answers is WHERE that hop lands, and a
        wide row's absolute density is small for reasons that say nothing about
        where it points. Ignores the causal window, per-query steering, and
        everything the residual stream does between passes.
        """
        mu, sigma, lam = self._atoms()
        reach = float((mu + 4.0 * sigma).max())
        span = max(64.0, self.cascade_hops * max(reach, 1.0))
        length = int(min(8192, 2 ** math.ceil(math.log2(span))))
        lattice = torch.arange(length, dtype=torch.float32, device=mu.device)

        kernel = (
            (
                lam[..., None]
                * torch.exp(_log_kernel(lattice, mu[..., None], sigma[..., None]))
            )
            .mean(dim=0)
            .sum(dim=0)
        )  # [length]
        kernel = kernel / kernel.sum().clamp_min(_EPS)

        spectrum = torch.fft.rfft(kernel, n=2 * length)
        acc = spectrum.clone()
        rows = []
        for _ in range(self.cascade_hops):
            hop = torch.fft.irfft(acc, n=2 * length)[:length].clamp_min(0.0)
            rows.append(self._sample_grid(hop))
            acc = acc * spectrum
        out = torch.stack(rows)  # [hops, G]
        return out / out.amax(dim=-1, keepdim=True).clamp_min(_EPS)

    def _sample_grid(self, curve: Tensor) -> Tensor:
        """Read ``curve`` (indexed by integer lag) at the display grid, linearly."""
        d = self.geom_lags.float().cpu().clamp(0.0, curve.shape[0] - 1.0)
        lo = d.floor().long()
        hi = (lo + 1).clamp_max(curve.shape[0] - 1)
        frac = d - lo.float()
        return curve[lo] * (1.0 - frac) + curve[hi] * frac

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, head_dim={self.head_dim}, atoms={self.num_atoms}, "
            f"causal={self.causal}, window={self.window_size}"
        )
