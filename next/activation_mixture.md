# Mixture of activations

Learn the nonlinearity instead of choosing it. `praxis/activations/mixture.py`.

## Where this came from

Two threads met.

The first is ours. `peer_dual` (2026-08) filled the GLU expert's empty linear
slot with a second activation, so every channel passed through two
nonlinearities in series - deeper, 34% more expensive per token, and the result
came back confounded. `peer_split` then asked the cheaper version of the same
question: keep the GLU's linear branch, and make the GATE heterogeneous instead -
experts 0-143 gate through Servant, 144-288 through swish, keyed on the expert
index so a bank row trains under one function class for the whole run. The
hypothesis both were testing is coverage: does the model want more than one
function class AVAILABLE, rather than composed?

The second is Manessi & Rozza, "Learning Combinations of Activation Functions"
(arXiv:1801.09403). Given base activations `F = {f_1 .. f_N}`, learn a point in

    conv(F):  sum_i c_i f_i(x),  sum_i c_i = 1, c_i >= 0
    aff(F):   sum_i c_i f_i(x),  sum_i c_i = 1

They report +3.01 top-1 for AlexNet on ILSVRC-2012 over fixed activations, on a
basis of identity / ReLU / tanh, and frame conv(F) as the regularized version of
aff(F): the convex hull cannot flip a base function's sign, so it preserves the
basis's monotonicity, while the affine hull can subtract one branch from another
and build shapes the basis does not contain.

## What was wrong with the split, and what was not

The IDEA was right. What was wrong was the LOCATION: the selection lived inside
a retrieval module, where it could only ever apply to PEER and could only ever
hold two functions. Everything else about it - discrete, permanent, keyed on the
expert index - is a legitimate hypothesis, and it is still runnable. `mix_split`
is the type that expresses it, and it still means what it meant.

What the move buys is that the discrete and continuous answers are now arms of
ONE experiment rather than two implementations:

|  | who picks the branch | when |
| --- | --- | --- |
| `mix_split` | the caller's index | frozen at init, permanent per row |
| `mix_gated` | the input value | per element, re-decided per token |
| `mix` / `mix_affine` | a learned scalar | one ratio for the whole model |
| `single` | nobody | there is only one |

The split is the mixture with one-hot coefficients frozen by index, so it is a
special case rather than a rival - which is exactly what makes the comparison
one variable.

## What was built

`ActivationMixture(activations, mode)` - a module that occupies an activation
slot and holds a bank of them. Anything that writes `ACT2FN[name]` gets it,
which is the whole point: it is the activation-level analogue of `ParallelHead`.

Types differ ONLY in where the coefficients come from:

| type | coefficients | what it is |
| --- | --- | --- |
| `mix` | `softmax(theta)` | the paper's `conv(F)` |
| `mix_affine` | `w + (1 - sum w)/N` | the paper's `aff(F)`, signs free |
| `mix_gated` | `softmax(slope * x + bias)`, per element | ours |
| `mix_split` | one-hot by an external index, per element | ours; was `act_alt` |

`mix_split` carries no coefficient parameters at all - the partition IS the key -
which is what makes it a clean control for the modes that learn one. The caller
hands each element a fraction in `[0, 1)` saying where it sits in whatever index
space the caller owns; the bank is cut into N equal segments. PEER passes
`expert / num_experts`. The fraction contract is what generalizes it: a head
axis, a depth, a codebook slot or a position all normalize the same way, and
none of them have to teach the mixture what their index space looks like. A
caller with no key (the dashboard's curve probe) gets the uniform blend rather
than an exception.

`mix_gated` is the version worth testing here. The coefficients are a function of
the pre-activation VALUE, so the blend is one distribution per element rather
than one number per model: the network learns which function class suits which
input REGIME - a non-periodic branch near zero and a periodic one out in the
tails, say - and re-decides per token. That is `peer_split`'s hypothesis with
the discreteness, the permanence and the locality all removed.

## One argument, one shape

Two rounds of opacity got fixed here, and they had the same cause: a choice the
model makes was recorded somewhere no reader could see it. First, a registry key
like `mix_harmonic` said a mixture was happening and nothing about what was in
it. Second - and worse - `ffn_type: peer_split` bound a feedforward to a fixed
activation, so a run whose experts gated through a servant/swish split reported
`activation: servant` on the dashboard's Arguments card. That card serializes the
launch namespace (`praxis/web/spec_data.py`), so anything decided after argument
parsing is invisible to it by construction.

So `--activation-type` now carries every activation choice, and it is always a
combination TYPE over a list of VALUES:

    activation_type:
      type: mix_split
      values: [servant, swish]

    activation_type: {type: single, values: [gelu]}
    activation_type: gelu                              # shorthand for the above

EVERY VALUE IS A GATE. `values` is the list of activations the gating position
draws from and `type` says how they combine there. A gated feedforward holds out
half its up-projection to multiply against, but that half is a STRUCTURAL choice
made by the feedforward, not an activation choice - which is why there is no
"gate" key. The one exception is `linear`, which activates that held-out half
(the old `dual_act` / `peer_dual` arm) and is named for the half it fills.

`single` is what makes this one shape rather than two: the ordinary
one-activation case is written the same way as a mixture, so nothing about a
config's shape changes when an arm adds a bank.

THE FALLBACK IS WHAT MAKES IT DECLARABLE MODEL-WIDE. `mix_split` partitions by
an index its caller supplies, and PEER's expert bank is the only place in these
models that has one. A type that needs an index and cannot get one falls back to
`values[0]`, so `{type: mix_split, values: [servant, swish]}` splits PEER's bank
and runs plain `servant` in the encoder, the heads and the controllers. That is
exactly what `-g` ran, which is what lets the split stay ONE change off `-e`
without needing a scope key.

The fallback has a sharp edge worth knowing: a lazily-shaped value that the
fallback never calls would still hold `UninitializedParameter` when the optimizer
walked `model.parameters()`, and raise there. `_materialize_unused` gives every
branch one no-grad forward the first time, so a bank whose first value is
parameter-free and whose second is not - a perfectly reasonable config - does not
crash the run.

WHICH BRANCH IS THE GATE. `GatedLinearMLP` computes `down(a * act(b))`, so `b`
is the gate in the SwiGLU sense (`Swish(xW) (x) xV`, the activated branch doing
the gating), and PEER names its banks the same way. The deleted
`DualActivationMLP` named them the other way round, calling the LINEAR half the
gate. That is worth recording only because the inverted naming outlived the file
and can make `config.activation` look like it was never the gate. It always was:
`git log -S swish -- praxis/dense/` shows swish first appearing in
`12efcdf0 dual_act` as `act_alt`, i.e. as half of the split, never as a gate
baseline.

`--activation` was also the last registry flag not ending in `-type`; it is
`--activation-type` now, mapped back to `config.activation` through the same
`arg_to_config_mapping` that already renames `ffn_type` -> `expert`. Experiment
YAMLs use `activation_type:`, and the old key raises rather than silently
outranking the flag.

## What moved out of the dense registry

`dual_act.py` went away, and so did four registry profiles. `dual_act`,
`peer_dual`, `peer_split` and `peer_mix` were each a feedforward bound to a fixed
activation choice; they are now declarations:

| was | is |
| --- | --- |
| `dual_act` | `glu` + `{type: single, values: [...], linear: gelu}` |
| `peer_dual` | `peer_glu` + `{type: single, values: [...], linear: gelu}` |
| `peer_split` | `peer_glu` + `{type: mix_split, values: [servant, swish]}` |
| `peer_mix` | `peer_glu` + `{type: mix_gated, values: [serpent, swish, linear]}` |

None of the arms are lost - they moved to where a card can see them. What
DENSE_REGISTRY holds now is feedforward STRUCTURE, which is the one thing a name
there can carry that a config value cannot. The combination types are their own
registry (`ACTIVATION_TYPE_REGISTRY`, docs/activation-types.md), which keeps
`ACTIVATION_REGISTRY` uniform: every entry there is something you can put in
`values`.

`linear` cannot be folded into a mixture, and that is not an inconsistency:
`act_linear(a) * act_gate(b)` multiplies two different PROJECTIONS, while a
mixture is `sum_i c_i f_i(x)` over one tensor.

Two small behaviour changes, both stated rather than hidden. The bank names
`servant` outright instead of inheriting `config.activation` - which for this
line is the same thing, since `-a` sets `activation_type: servant`. And the
segment boundary is a floor of the fraction rather than
`index >= num_experts // 2`, which moves exactly one expert on an odd bank - 729
splits 365/364 instead of 364/365, the same rounding the old code declined to
assert on. At `-u`'s aligned 676 it is exactly 338/338.

## What to watch

- `activation_mix_routing` is the metric that matters, and it exists because of
  the Servant chirp: a gate with a large but CONSTANT preference is a static
  blend wearing a router's clothes, and every magnitude metric reads it as
  healthy. Dispersion across elements is the only thing that separates them.
  Zero means no routing is happening, whatever the shares say.
- `activation_mix_entropy` at 1.0 forever means the model has no preference and
  the bank is doing nothing a single averaged function would not do. Falling to
  0 means it collapsed to one branch - which is a real answer (it says a fixed
  activation was correct) and cheaper to read than a bpb curve.
- `activation_mix_share_*` per branch. If `serpent` wins outright everywhere,
  the periodic prior is the whole story and this thread closes.
- BPB on all three axes. The `mix_gated` arm is NOT parameter-identical to a
  plain `peer_glu` (2N
  gate scalars per mixture, plus Serpent's spectrum now materialized inside the
  bank), but the difference is rounding error; a token-axis-only win should be
  suspected of being the batch governor, exactly as it was on -f.

## Known wrinkle, inherited not introduced

Inside PEER the activation is applied to `[b, n, h, k]`, so any per-feature
activation binds its parameter vector to the RETRIEVAL RANK axis, not to
features. Serpent under a `mix_gated` expert slot gets `k=8` frequencies indexed by "how
good was this expert's score", which is not a meaningful axis to be
per-parameter over. This predates the mixture - `config.activation: servant`
with `ffn_type: peer_glu` already did it - and fixing it would change every
existing PEER run, so it is recorded rather than repaired. It is also an
argument for `swish`/`linear` carrying the load in that bank.

## Next

- The obvious missing arm is a LEARNED keyed table: coefficients per segment
  that train, instead of a frozen one-hot. That is the honest middle between
  `keyed` and `gated` - the assignment stays a property of the bank row, but the
  row can change its mind. Not built; nothing yet says the frozen version is the
  binding constraint.
- A channel-split `dual_act` is now unnecessary: name a `mix_*` profile as its
  `activation` or `activation_gate` and the same question runs without PEER.
- The bank can hold parametric members (`snake`, `sinlu`, `prelu`) or other
  mixtures. Nothing tests a deep bank yet; the cost is linear in `N` and every
  branch is evaluated densely, so a wide bank is not free.
- KAN is the limit of this idea (a learned activation per EDGE) and is already
  in the registry, so if a shared learned activation pays, the scale-up path
  exists and the comparison is already available.
