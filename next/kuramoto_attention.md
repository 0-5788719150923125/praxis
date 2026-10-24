# Kuramoto Attention: attention as a synchronization step

> Status: **built, first run pending** (2026-10-23). `byte-latent-i` =
> `-g` + `attention_type: kuramoto` + `encoding_type: rope_learned_rates`.
> Paper: Nunley 2026, "Kuramoto Attention: Synchronizing Self-Attention on the
> Torus", arXiv:2606.11585 (read in full; no code released). Grew out of
> [synchronicity.md](synchronicity.md), reading 3 (binding by synchrony).

## The mechanism

Each token is a bank of phase oscillators. Per layer:

- **Score**: `s_tu = tau/sqrt(k) sum_j g^q_tj g^k_uj cos(theta_tj - theta_uj + omega_j (t - u))`.
  The gates are `softplus(W psi(theta))` normalized to mean 1, where
  `psi = (cos, sin)`. This is exactly a RoPE dot product of the gated lifts,
  so it runs on the stock attention kernel.
- **Value**: the raw phases. `G_t = sum_u A_tu e^{i theta_u}` is the
  attention-weighted circular mean.
- **Update**: the tangent projection `a = -sin(theta) Re G + cos(theta) Im G = sum_u A_tu sin(theta_u - theta_t)`
  is exactly Kuramoto coupling for fixed `A` (the paper's Lemma 2). It is
  scaled by a signed gate `v(theta)` (positive attracts, negative repels) and
  bounded by `r tanh(|x|/r) x/|x|` with a learned `r`.

The paper's claims are mostly about legibility, not loss:
- **Loss:** about +0.02 BPC at 1M parameters and about -0.01 at 5M against
  matched RoPE+SwiGLU.
- **Local order** (attention-weighted coherence) stays near 1.
- **Global order** falls with depth, from 0.98 to 0.56: tokens bind into
  groups without collapsing.
- **Learned natural frequencies** fall with depth: early layers couple
  short-range, later layers long-range.

Ablations at 1M, in BPC:

| change | cost |
|---|---|
| learned value projection instead of the raw-phase mean | +0.25 |
| low-rank value transport | +0.33 |
| remove the FFN | +0.27 |
| remove the value gate | +0.11 |
| unshare the metric gates across layers | +0.05 |

The narrow, coordinate-separable value path is what it rewards.

## How it sits in Praxis (no special block)

- `praxis/attention/phase.py`: `PhaseProjection` (the qkv seam) and
  `TangentStep` (the output seam). `SelfAttention(projection=, output=)` takes
  any stage classes, and the `kuramoto` registry entry is
  `partial(SelfAttention, projection=PhaseProjection, output=TangentStep)`.
  Profile keywords now reach `patch_config`, so a stage can correct the head
  layout.
- **Phase reader**: `theta_j = alpha_j * x_j`, a learned per-coordinate
  frequency, as in Serpent. The step is divided by `alpha` with Serpent's
  rectified inverse. `head_size` stays the width, and the head count becomes
  `hidden_size // (head_size // 2)`, with one query per head.
- `praxis/encoding/learned_rates.py`: `rope_learned_rates` has a learned rate
  per rotary pair per KV head, with geometric init and a per-depth shift.
  Any attention can use it.
- **Metrics**: modules can declare `per_layer_metrics`.
  `collect_attention_metrics` logs those as `<key>_layer<i>`, one line per
  layer on a shared chart, because the depth profile is the signal. Stats are
  written to a buffer in-graph (no host sync, no graph break). The ghost mass
  comes free from FlexAttention's log-sum-exp.

**Deviations from the paper:**
- Gates are per layer. The paper shares them, and unsharing cost it +0.05.
- Ghostmax stays on, so a token can decline to couple.
- Phases are read through the block's pre-norm, not a pure torus state. The
  direction is exact coupling for the layer's phases, but the size of the move
  on the next layer's phases is learned.
- The FFN is unbounded (PEER GLU + Serpent).

## What to read on -i

Dynamics tab, "kuramoto" section:

| chart | paper's shape | what would be interesting here |
|---|---|---|
| Local Phase Order | ~0.9-0.99, slight fall with depth | stays high = selection and synchronization agree |
| Global Phase Order | falls with depth | flat across layers = no differentiation by depth |
| Repelling Oscillators | not reported | where it settles from ~0.5, and whether it rises where global order falls |
| Decoupled Mass | n/a (no ghost) | how much coupling tokens decline |
| Phase Step / Step Bound | n/a | bound near 1 = the radius never binds |
| Mean Rotary Rate | falls with depth | the depth-to-timescale trend reappearing unprompted |

At init, global order is about 0.61. That is `e^{-1/2}`, for unit-variance
input read at `alpha = 1`.

## Open after the first run

- **The cognitive test the paper proposes.** Do high-coherence coalitions line
  up with real units (words, lines, code blocks)? That needs a checkpoint probe
  of local order against byte position in the word or line. It is not a
  training metric.
- **Arms:**
  - plain `rope` (isolates the learned rates);
  - shared gates across layers (the paper's reference);
  - learned higher-harmonic coupling, `sum_n c_n(x) sin(n dtheta)`, which is
    gap 2 in [magnetism.md](magnetism.md).

## Found on the way: Praxis RoPE used half its spectrum (fixed 2026-10-23)

`RoPE._compute_rope_embeddings` built GPT-J's interleaved cos/sin cache
(`stack(...).view`, which is `repeat_interleave`), but
`_apply_rotary_pos_emb` pairs dimensions Llama's rotate-half way
(`chunk(2)`). That mixes the two conventions. Measured at head_dim 8, the
four pairs turned at `-w0, +w0, -w1, +w1` instead of `w0, w1, w2, w3`. So
every RoPE, HoPE and ArcHoPE run used only the faster half of its bands, each
twice with opposite signs, and the slowest wavelength was about
`2 pi sqrt(theta)` rather than `2 pi theta`.

**The fix:**
- **Checked against the references:** transformers' Llama `rotary_emb` +
  `apply_rotary_pos_emb` (cache `cat(freqs, freqs)`) and GPT-J's interleaved
  `rotate_every_two`.
- **Cache:** it is now in rotate-half order. Output matches transformers'
  `apply_rotary_pos_emb` exactly; the old layout differed from it by up to
  6.3.
- **Odd head_dim:** the unpaired slot now passes through unrotated.
- **Tests:** in `tests/encoding/test_rope.py`.

Every run before this date trained on the half-spectrum version, and resuming
one picks up the fix. `rope_learned_rates` never had the bug.
