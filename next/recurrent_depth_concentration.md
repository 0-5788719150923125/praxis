# Does the deviation concentrate as recurrent depth increases?

> Status: **testable, not yet run** (2026-06-30). Extracted out of
> [theory_of_everything_and_nothing.md](theory_of_everything_and_nothing.md),
> which became a narrative piece and shouldn't be the place this prediction
> lives. Sibling to [harmonic_koopman.md](harmonic_koopman.md) (the
> wind-up/release-to-basis reading this borrows) and
> [oscillatory_axes.md](oscillatory_axes.md) (the recurrence axis it proposed
> testing first, before the float-precision detour got parked).

## The prediction

`sec:watchmaker` in the paper reads recurrent depth as a wind-up that releases
back to the stable basis - the state stays close to the standing harmonic
shape, punctuated by sparse, discrete jumps where a geometry is briefly
expressed and then reverts. `sec:harmonic`'s "outer representation" subsection
(`body.tex`) goes further and predicts the deviation itself is *concentrated*:
most positions/features do null work, a sparse scatter carries the departure
from the standing shape - checkable via `harmonic_delta_norm`, already logged.

Neither claim has been read *across recurrent-depth steps specifically*. The
clean prediction: **as recurrent-depth step increases, the deviation should
become more concentrated** - not necessarily smaller in raw magnitude, but
sparser, with more of its energy sitting in fewer positions/features - tracking
the same wind-up-then-release-to-basis reading one level down, from "across
training" to "across one forward pass's depth loop."

## Why this, and not the float-precision framing

An earlier framing of this same intuition tried to locate the mechanism in
floating-point precision itself - the model "targeting" specific representable
float values, error boundaries as the carried signal. That is mechanically the
idea `oscillatory_axes.md` already tried and parked on 2026-05-29, for two
reasons that still hold: reducing all representable floats to a hash destroys
the very structure that would be interesting (no axis), and the bucket grid is
fixed by IEEE-754 - there is no differentiable path from "which bucket a value
rounds into" back to the loss, so gradient descent has no way to select for
it. What *is* real and already cited (`sec:constructor`, `body.tex`) is a
weaker, correlational claim: precision artifacts (measurement-induced
rounding) concentrate on the exponential edge, because that is where the field
is already most plastic - the same narrow, sensitive operating band drives
both. That correlational reading, not a mechanism claim about targeted
buckets, is the honest anchor for "precision reveals where change happens."

## The test

Both diagnostics already exist for the static spectrum (`concentration()`,
Hoyer sparsity of the amplitude grid; `harmonic_delta_norm`, the deviation's
magnitude and its concentration across positions/features). The only new work
is reading them **per recurrent-depth step** instead of once at the end of the
loop. A flat or *decreasing* concentration across depth refutes the
"twist contracts toward the stable field" reading; the paper's own
falsification criterion for `sec:watchmaker` (a state that drifts and never
returns to the basis) is the same test, read one level down. Cheap to run -
the instrumentation is the hard part of most diagnostics, and it already
exists here.

## Prior-art anchors

`sec:watchmaker`, `sec:harmonic` ("The outer representation"), `sec:scaling`
("The exponential edge"), `sec:constructor` (all `research/body.tex`);
[harmonic_koopman.md](harmonic_koopman.md); the float-precision parking and
its objections, reused verbatim above, from
[oscillatory_axes.md](oscillatory_axes.md); the precision-as-addressing rescue
in [hash_gated_anchor.md](hash_gated_anchor.md).
