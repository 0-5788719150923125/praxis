# Periodicity Generalization: An External Prior, and the Objection Worth Making

> Status: **external result assessed 2026-08-30, one testable claim extracted.**
> Source: *Do Transformers Have the Ability for Periodicity Generalization?*
> (arXiv 2601.22690v1), code at `github.com/gtxygyzb/periodicity_generalization`.
> Companion to [collapse_regime.md](collapse_regime.md),
> [harmony.md](harmony.md), [harmonic_koopman.md](harmonic_koopman.md).

## The one line

The paper is sound work whose title overreaches its evidence - and the honest
reply is not "wrong architectures", it is **one specific mechanism plus a
falsifier**, because their benchmark is released and we could just run it.

## What it actually found

Composite-periodicity benchmark ("Coper") with two OOD splits: **Hollow**
(gaps inside the training range) and **Extrapolation** (beyond it).

| model | ID | Hollow | Extrapolation |
|---|---|---|---|
| Transformer | 94.7 | 29.8 | 22.6 |
| **FANformer** (Fourier network) | 89.5 | **41.7** | **19.4** |
| Mamba | 49.5 | 43.1 | 45.3 |
| RWKV | 64.4 | 62.6 | 60.4 |

**The line that matters here.** FANformer is the closest published analogue to
what this repo builds. A Fourier inductive bias bought **+12 on Hollow and -3 on
Extrapolation**: it fills holes inside the training range and does nothing to
extend past it. That is the strongest external evidence available that harmonic
bias and extrapolation are not the same purchase, and it should be held as a
prior rather than argued away.

Scaling (3 -> 7 layers) took the Transformer 29.8 -> 57.2 on Hollow but only
22.6 -> 32.7 on Extrapolation. Depth mostly buys interpolation.

## Two things to discount

**RWKV's win is mostly underfitting.** At 64.4 ID it never fit the task, so its
flat 62.6/60.4 is uniform mediocrity, not extrapolation. The "Avg." column
rewards models that failed to fit. What the table really shows is that the
ID/OOD *gap* tracks how well a model fit at all - a much weaker claim than the
architecture ranking implies.

**The "formal proof" proves less than the abstract.** The commutation argument
shows RoPE's shift-invariance does not *automatically* deliver rule periodicity.
It does not show a transformer cannot represent it - with MLPs it is a universal
approximator on bounded domains, so the real claim is about what is *learned*.
And "in general, the two group actions do not commute" is asserted, not derived.
Motivation, not impossibility.

## The distinction worth stealing

**Sequence periodicity**: the input repeats with period T. RoPE handles this;
they prove it and we already exploit it (ArcHoPE, learnable theta).

**Rule periodicity**: the *operation* repeats, `R_{a+T} = R_a`. Their claim is
that position-encoded periodicity gives you the first and not the second.

That is a sharper frame than "harmonic models should extrapolate", and it is the
one to design against.

## The objection, made properly

"We would blame their architectures" is the move that makes a program
unfalsifiable, and it is the same move this repo criticises elsewhere. Here is
the version with a mechanism in it:

**Weight-shared recurrence is rule periodicity by construction.** `R_{a+T} =
R_a` is a literal description of applying the same block at every pass.
Praxis runs `num_layers: 1, depth: 6` - one physical block, reused. Every model
in their table is a stack of *independent* layers, where the rule at position a
and the rule at position a+T are different parameters that must separately learn
to agree. We do not have to learn that agreement; it is an identity.

If the paper's diagnosis is right, that is exactly the axis their architectures
lack and this one has. Which makes it a prediction, not an excuse.

## The falsifier

Their code is released and the benchmark is small. **Run Coper on a
weight-shared recurrent praxis config against a parameter-matched stacked
transformer.**

- Recurrent >> stacked on Extrapolation: the rule-periodicity mechanism is real,
  the paper's negative result is architecture-specific in exactly the way
  claimed, and this is the first genuinely novel thing this line has to say.
- Recurrent ~= stacked: weight sharing does not buy rule periodicity, the
  objection was cope, and the FANformer prior stands. Cheap to learn, and worth
  knowing before more harmonic capacity gets built.
- Both fail at any scale we can run: the small-model regime is the confound and
  Coper says nothing about us either way.

Caveat to set now: their scaling result cuts *against* us on size. If depth
3 -> 7 helps and we run 3.2M parameters at depth 6 with one shared block, we
should expect to sit near the bottom of their curve. A weak absolute score is
not evidence for the mechanism; only the **gap against a parameter-matched
stack** is.

## Standing note

Nothing here is a defence of the harmonic line. The FANformer number is the
cleanest external datapoint we have and it points the other way. The only thing
this note claims is that there is a specific, cheap, pre-registered way to find
out whether the objection has anything in it - and until that runs, the prior
should be theirs, not ours.

## What the released code actually does (read 2026-08-30)

`generate_periodic_data.py`, the `2seq_add` task:

- Sample two periods `p1, p2` from `PERIOD_RANGE` (2..16). Build two random
  sequences of those lengths, tile each to their LCM, sum elementwise mod 10.
- Render as a string: `seq1 + "+" + seq2 + "=" + seq_sum`, then train
  next-token prediction on it (scored after the `=`, since the prefix is random
  by construction).
- **Train** excludes two regions of the `(p1, p2)` grid.
- **Hollow** = `HOLLOW_SET = {8,9,10,11}` squared - a hole punched in the middle.
- **Extrapolation** = `BORDER_PAIRS`, i.e. `p or q in {2,3}` **or** `{15,16}`.

### Two things that only show up in the code

**"Extrapolation" is a border RING, not a direction.** It includes periods
*shorter* than anything trained (2, 3) as well as longer (15, 16), pooled into
one number. Those are different asks - short periods are closer to a degenerate
constant sequence, long ones require composing beyond the trained LCM range.
Reporting them as a single column makes the headline weaker than it reads.

**The measurement is inseparable from training on the task.** Coper reports
generalization *within* a task the model was trained on: their models learn
composite periodicity from the training grid and are then tested on held-out
regions of that same grid. A model trained on code and prose is at chance on all
three splits, and the ID:Hollow:Extrapolation ratio is then noise over noise.

## So: this cannot be a running metric on our normal runs

Not without training on Coper, which means either a dedicated run or injecting
synthetic data into the mixture - and the latter changes the thing being
measured. There is no version of this that rides along on a code-and-prose run,
because the benchmark has no signal unless the rule is in the training
distribution.

Two ideas that looked free and are not, recorded so they are not re-proposed:

- **Depth extrapolation** (evaluate at loop counts above the trained range).
  Invalid: KL halting trains *toward fewer* loops, so by depth 7+ the latent has
  already converged and the input barely changes. It measures a fixed point, not
  an extrapolation.
- **Sequence-length extrapolation.** Invalid as posed: the packer concatenates
  unrelated documents at arbitrary boundaries, so a longer sequence is not a
  longer *document* and there is no well-defined "correct" continuation to score
  extrapolation against.

## What remains

Coper is an **experiment, not a metric** - the standalone falsifier already
described above (weight-shared recurrent vs parameter-matched stack, on their
released code). Small, separate, contaminates nothing. That is the only honest
way to use it, and the rule-periodicity question is worth one small run.

## If it were built as a praxis task dataset (design, not built - 2026-08-30)

Training on Coper is what unblocks the metric, and the pattern already exists:
`SyntheticPrintDataset` generates on the fly with no files. The shape would be a
`PraxisSampler` subclass + a `DATASETS` entry + a validation hook. Parked, not
built. Recorded so the traps do not have to be rediscovered.

### The confound that would sink it, and it is ours alone

**`patch_size: 8` aliases the periodicity.** Their models are char-level; the
abstractinator sees latents at 1/8 resolution. A period-3 digit sequence has no
clean patch-level representation - it smears across patch boundaries at an
effective patch-space period of `LCM(p, 8) / 8`. Over their `PERIOD_RANGE` 2..16:

- aligned (divide or multiply 8): **2, 4, 8, 16**
- aliased: **3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15**

It lands unevenly on their splits - `HOLLOW_SET {8,9,10,11}` has one aligned
member, `BORDER {2,3,15,16}` has two. So a poor extrapolation score could mean
"patching destroyed the signal" rather than "the architecture cannot
extrapolate", with no way to separate them after the fact.

**The fix is free and converts the confound into the result:** report accuracy
split by whether `p1, p2` divide the patch size. If aligned periods generalize
and aliased ones do not, that is a finding about the patcher - and it is an axis
their paper structurally cannot see, since char-level models have no patch grid.
That may be worth more than the original metric.

### Two more traps

**Length.** `2 * LCM(15,16) = 480` plus prefix, so the longest examples run ~500
bytes against rows of 64-512. Long pairs truncate, cutting off the answer.
Either cap `PERIOD_RANGE` (which changes the benchmark, and should be declared)
or emit an example-length metric so truncation is visible rather than silent.

**Packing.** An example split across rows loses its answer entirely. The sampler
must guarantee one example per row, or the carryover path corrupts the eval
quietly.

### Design notes

- Enforce the train/test exclusion **inside the generator** - training draws
  never touch `HOLLOW_PAIRS` or `BORDER_PAIRS`. A bug here makes every
  downstream number worthless, so it is the first thing to test, before any
  metric is reported.
- The `DATASETS` entry needs an **explicit `task_type`**; a type-only entry
  silently becomes `PRETRAIN` at 10x weight and would swamp the mixture
  ([[reference_dataset_task_type_fallback]]).
- **Let `sampler_mode: tasker` learn the mixture weight** rather than setting
  one. It already learns per-task difficulty -> sampling weight, so the fraction
  does not become a tuned constant.
- **Split their Extrapolation column.** `BORDER_PAIRS` pools periods *shorter*
  than trained (2, 3) with longer ones (15, 16). Reporting `extrap_low` and
  `extrap_high` separately costs nothing and is strictly more informative than
  their headline.

### Standing caution

This is the only route to a valid extrapolation metric found so far, and it is
also the most invasive: it puts synthetic data in the training mixture of every
run that carries it. Worth it only if the extrapolation question becomes the
question, rather than one of several.
