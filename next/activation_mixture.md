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
expert index - is a legitimate hypothesis, and it is still runnable. `keyed` is
the mode that expresses it, `mix_split` is the profile, and `peer_split` still
means what it meant.

What the move buys is that the discrete and continuous answers are now arms of
ONE experiment rather than two implementations:

|  | who picks the branch | when |
| --- | --- | --- |
| `keyed` (`peer_split`) | the caller's index | frozen at init, permanent per row |
| `gated` (`peer_mix`) | the input value | per element, re-decided per token |
| `convex` / `affine` | a learned scalar | one ratio for the whole model |

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

## The bank is never baked into the name

A registry key like `mix_harmonic` says a mixture is happening and nothing about
what is in it, which is the same opacity the whole refactor was meant to remove.
So an activation is declared one of two ways and `build_activation` resolves
both::

    activation: gelu

    activation:
      type: mix_split
      values: [servant, swish]

Every config-driven site goes through that one function, which also retires the
two call styles that had grown up side by side (`ACT2FN[name]` and
`ACT2CLS[name]()`, the second of which quietly mishandled the `(class, kwargs)`
tuples transformers registers for a few of its own entries). A bank entry may
itself be a spec, so a mixture can hold a mixture with no special case.

The four `mix_*` names are the only registry entries that cannot stand alone;
`TYPED_ACTIVATIONS` is what says so, and using one bare raises at declaration
time with the spec form in the message rather than at model-build time.

`dual_act.py` went away in the same pass, for a related reason: it was a
`GatedLinearMLP` with an activation on the value branch and nothing else, so it
is now `partial(GatedLinearMLP, activation_value="gelu")`. The value branch is a
second SLOT, not a second entry in one slot - `act_v(a) * act_g(b)` over two
different projections - which is why a mixture cannot express it and the second
kwarg stays.

**Coefficients are scalar, not per-channel**, and that is deliberate. The paper
parameterizes its combination as a 1x1 convolution, i.e. one coefficient vector
per channel. That only means something if the tensor's last axis is a feature
axis, and here it often is not - PEER applies its activation to `[b, n, h, k]`,
whose last axis is retrieval RANK. Per-channel weights would bind silently to
the wrong quantity. Scalar coefficients keep the module a pure elementwise
`R -> R` function, which is what lets it stand anywhere; `gated` recovers
element-level resolution by reading the input's VALUE instead of its POSITION.

**Init is uniform**, against this codebase's usual habit of starting a learned
blend as the thing it replaces (Servant is exactly Serpent at init; the
prismatic field is identity at init). Those wrap a known-good baseline; a
mixture has none, and biasing it toward branch 0 pre-loads the answer to the
question it exists to ask. Uniform also puts `activation_mix_entropy` at exactly
1.0 on step 0, so any movement is signal.

The two banks in use live in `praxis/dense/__init__.py` next to the profiles
that name them: `SPLIT_BANK = [servant, swish]` (what the original split ran) and
`HARMONIC_BANK = [serpent, swish, linear]` - periodic, non-periodic,
pass-through, the pass-through being what lets a feature decline both. So
`peer_split` is `peer_glu` with `{type: mix_split, values: SPLIT_BANK}` and
`peer_mix` is the same with `mix_gated` over `HARMONIC_BANK`. The `*_even` twins
that used to sit beside them are gone: the key rounding they existed to pass is
requested by PEER now (`praxis/transforms/alignment.py`), so there is one entry
per arm.

Two small behaviour changes in `peer_split`, both stated rather than hidden. The
bank names `servant` outright instead of inheriting `config.activation`, so the
profile means the same thing under any `--activation` (it matches
abstractinator-a, which is what this line runs, so `-g`, `-q` and `-u` are
unaffected). And the segment boundary is a floor of the fraction rather than
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
- BPB on all three axes. `peer_mix` is NOT parameter-identical to `peer_glu` (2N
  gate scalars per mixture, plus Serpent's spectrum now materialized inside the
  bank), but the difference is rounding error; a token-axis-only win should be
  suspected of being the batch governor, exactly as it was on -f.

## Known wrinkle, inherited not introduced

Inside PEER the activation is applied to `[b, n, h, k]`, so any per-feature
activation binds its parameter vector to the RETRIEVAL RANK axis, not to
features. Serpent under `peer_mix` gets `k=8` frequencies indexed by "how
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
