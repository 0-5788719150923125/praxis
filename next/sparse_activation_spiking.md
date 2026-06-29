# Near-identity dynamics and a sparse, event-driven reading

Status: conjecture + marked voice (2026-06-29). The notion: because of the
attention sinks and the dominant static spectrum, the model is incentivized
to *mostly not transform* across inputs - the per-input computation is sparse and
event-like, against a shared standing shape. Pairs with
[[project_harmonic_latent_koopman]] (states carry a velocity; Koopman),
[[project_harmonic_amp_modulation]] (the input-conditional delta + its L2
penalty), `research/body.tex` "The outer representation", and the conclusion's
moving-point coda. Lives one register out from the paper - a paragraph went into
the outer-representation section, the rest stays here.

## The grounded spine (defensible from the architecture)

- **Silent heads = identity pass-through.** Ghostmax lets a head spend its mass
  on the positionless zero ghost and contribute nothing downstream (the paper's
  own "dampening valve"). A head that goes quiet is an identity on the residual
  stream. So sparsity of *active* heads is a real, designed-in possibility.
- **Input-conditional variation is small by construction.** The field is
  `a_static + Delta_phi(context)`, Delta zero-initialized and L2-penalized, the
  static baseline kept dominant. So across two inputs the field is dominated
  by the *same* corpus rhythm; what differs is a small delta. "Every sequence
  returns a shape that looks mostly no different from any other" is the accurate
  reading of *biased to the static spectrum* - the transform is nearly
  input-independent, not literally identity (the static field is itself a real,
  substantial transform).
- **Temporal hold = velocity + events.** The smoothness prior asks the field at
  position t to predict the field at t+1: hold the shape over time. A hidden
  state mostly carries its geometry forward (enters with a velocity, the Koopman
  reading), and is occasionally kicked off it by a large input-conditional
  deviation. That cadence - mostly coast, rarely spike - is event-driven.

## The leap (voice, marked - keep out of the paper as mechanism)

- **Spiking-neural analogy.** Sparse, large deviations on a near-constant base
  *resemble* spiking activity. This is a point of contact with biological
  plausibility, **motivation not mechanism**. The architecture is not an SNN; do
  not claim event-driven hardware semantics, refractory dynamics, or that the
  deviations are literal spikes.
- **CALM voters raising their own events.** The wildest extension: clusters of
  decoded votes (CALM's temperature-as-agreement) could *raise* activations and
  feed them back to the sparse model - a closed loop where consensus emits the
  events the field then responds to. Unimplemented, no design, no metric. A
  direction, not a result.

## The "K^2 state spaces" point (clarify, do not ship as-is)

His phrasing: "periodic over harmonics that produce K^2 distinct state spaces."
The amplitude grid is `F_t x F_d` (~K^2 cells), so K^2 is the *grid size*. But the
paper's expressivity claim is **exponential in K** (interference across
components: log N components address N configurations), not K^2. Keep these
separate - writing "K^2 distinct states" into the paper would *undercut* the
scaling conjecture. The 2D grid having ~K^2 cells and the field addressing
exp(K) configurations are different statements.

## What would confirm it (falsifiable, uses metrics we already have)

- `harmonic_delta_norm` = rms(Delta)/rms(a_static) stays small: the per-input
  transform is mostly the static baseline.
- The delta is *concentrated* - sparse across positions and features rather than
  diffuse. This sparsity is the spiking-like prediction; a diffuse delta refutes
  it. (Needs a concentration/Hoyer-style read on Delta, not just its norm - the
  norm is magnitude, sparsity is shape.)
- Fraction of heads at/near the ghost (silent-head rate) is high and the active
  set is small and input-dependent.

## Where it landed in the paper

A single paragraph in "The outer representation" (already the section that marks
itself the least-settled, conjecture register), framed as: silent heads +
small delta => near-input-independent transform => sparse deviations carried
forward with a velocity => event-driven reading; SNN as motivation; CALM
feedback flagged as an open question; measurable via `harmonic_delta_norm` + its
concentration. The K^2 framing and the full CALM-feedback loop stay here.
