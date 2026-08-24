# Position as a query: the field-readout draft head

> Status: **not started** (2026-08-23) - design note only, nothing built.
> Gated on the $k > K$ unroll recorded in
> [paper_self_criticism.md](paper_self_criticism.md) (entry 1): if the existing
> serpent cell shows a cliff at the trained boundary rather than graded decay,
> none of this is worth building. Sibling to [mtp_curve.md](mtp_curve.md),
> which owns the "MTP turns the vector into a curve" reading this note takes
> literally, and to [harmonic_koopman.md](harmonic_koopman.md).

## Where this came from

A conversation about seq2seq and the Library of Babel: if a single vector can
determine a whole sequence, why not treat the classes as one enormous linear
hash space and predict a position along it?

The 1D version is a dead end, and the reason is worth keeping because it
determines the shape of the repair. Enumerating discrete sequences along an
ordinal axis destroys local geometry - index $n$ and $n{+}1$ differ in the last
symbol, index $n$ and $n + |V|^{T-1}$ differ in the first - so the loss surface
in that coordinate is a devil's staircase and the gradient carries no usable
information. The only ordering with good local geometry is the model's own CDF,
which is arithmetic coding, which is the model you were trying to learn. It is
circular by construction.

What survives the objection is the same idea with the dimensionality left in:
**keep the address high-dimensional, and move the sequence axis out of the
address entirely, into the decoder as an argument.**

## The idea

Today's draft head answers "what is the byte at depth $k$" with a per-depth
module (or a shared cell unrolled $k$ times). Depth is a slot. The proposal is
to make it a *coordinate*:

- The trunk state produces coefficients $a$ over the frozen harmonic basis.
- The byte at offset $\tau$ is read by evaluating that field at absolute
  position $t + \tau$ and passing the result through the existing lm_head.
- $\tau$ is an argument, not an index into a table of modules.

Nothing about the address grows with $K$, because $K$ is not encoded in it.
$K$ becomes how many times you choose to evaluate.

This is what makes the paper's second discriminator measurable by construction
rather than by arrangement: a fitted signal has values where no sample was
taken, so $\tau > K$ is a well-formed query even though nothing supervised it.
A bank of $K$ independent modules has no module at $K{+}1$ - the query is
undefined, not merely untrained.

## Why the machinery is already mostly there

Checked 2026-08-23, and it is further along than expected:

- `HarmonicField._eval_field(scaled, seq_len, device, offset=0)`
  (`praxis/heads/harmonic.py:814`) already evaluates the field at
  `offset .. offset+seq_len-1`. The field is anchored to ABSOLUTE position;
  the offset exists for cached-decode continuation.
- `_phase_table` (`harmonic.py:833`) slices a precomputed `[T, F_t]` buffer,
  and its past-period fallback (`harmonic.py:849`) already builds phases from
  a **float** position tensor. Continuous $\tau$ needs that path generalized,
  not invented.
- `amp_modulation` in `("input", "pure")` (`harmonic.py:596`) already projects
  pooled hidden states to envelope coefficients through `amp_input`. That is
  the "coefficients come from the trunk state" half, already built and already
  the variance axis the paper's manifold section talks about.

## The build

**0. The gate (no new code).** Unroll the existing serpent cell past $K$:
`praxis/heads/mtp/rnn.py` is one gated cell, and `depth_embed`
(`rnn.py:57`) is `torch.zeros(num_depths, ...)`, so holding the signature at
zero beyond the trained window is the cell's own default rather than an
invention. Score against true bytes at $k \in [K{+}1, 2K]$, plot accuracy
against $k$. Graded decay across the boundary means proceed; a cliff at $K$
means stop and demote the address language in the paper.

**1. Generalize the query.** `_phase_table(seq_len, device, offset)` takes an
explicit position tensor instead of deriving `arange(offset, offset+seq_len)`.
One signature change; fractional $\tau$ comes with it for free.

**2. `praxis/heads/mtp/field.py`.** One readout, `(trunk state, offset τ) ->
logits`, parameters $O(1)$ in $K$ and strictly fewer than any per-depth bank.
It slots into the existing bank contract (`prepare_inputs`,
`training_metrics`, per-depth losses) the same way vear and serpent_rnn do.
Note the wart while you are in there: `MTP_REGISTRY`
(`praxis/heads/mtp/__init__.py:27`) holds only the per-depth module types -
bank types are branched by string at `__init__.py:129-141` and appended by
hand to the CLI choices at `praxis/cli/groups/architecture.py:426`. A third
bank is the moment to make the registry hold banks too, rather than the fourth.

**3. Train on $\tau \in \{1..K\}$, evaluate on $\tau \in \{1..2K\}$.**
`mtp_draft_acc_d{k}` already exists and already charts; it just gets rows past
$K$. The extrapolation curve stops being an experiment and becomes something
the run reports.

## What it buys

- The paper's discriminator becomes a logged metric instead of a promise.
- Draft width becomes a runtime dial rather than an architecture constant -
  which is what the adaptive-width work already wants (`draft_width` at
  `mtp/__init__.py:180` currently clamps to `num_depths` because there is
  nothing to evaluate beyond it).
- Fractional $\tau$ becomes *defined*. Meaningless for bytes; not meaningless
  for the codec latent, where "query between patch boundaries" is a real
  question and nobody has an answer.
- Fewer parameters than every existing bank.

## What it likely costs

Stated in advance so a bad result is not reinterpreted afterwards:

**It will probably lose to independent per-depth banks on in-window
accuracy.** Forcing every offset through one shared field is a constraint the
free bank does not have. The payoff is the measurement and the parameter
count, not a leaderboard win, and the note should be judged on that.

**The address bandwidth may be the binding constraint.** `amp_input` is
`nn.Linear(D, amp_K)` where `amp_K = F_t + F_d` - a low-rank *envelope* over
the frozen spectrum, not the full `F_t x F_d` grid. If the field-readout draft
underperforms, check whether the conditional channel is simply too narrow to
carry a $K$-byte future before concluding anything about the idea.

**All $K$ drafts share one trunk state $h_t$ and differ only by positional
modulation.** That is the point (it is what makes $\tau$ a coordinate), and it
is also the most likely source of a flat profile across depths. Instrument for
it: a field readout whose accuracy is identical at every $\tau$ has collapsed
to a position-independent guess, which is the failure mode to watch, and it is
the direct analogue of the flat-signature collapse the serpent cell already
instruments.

## Prior art

Nobody appears to do this for text. The structure is well established
elsewhere, which is the useful part - the pieces are known to work, and the
open question is confined to the domain.

- **DeepONet** (Lu, Jin, Karniadakis 2021): branch net encodes the input into
  coefficients, trunk net evaluates at any query coordinate, output is their
  inner product. Structurally identical to the proposal.
- **Fourier Neural Operator** (Li et al. 2021): learns the kernel in the mode
  basis - the Koopman claim as an architecture rather than an interpretation.
- **Functa** (Dupont et al. 2022): the data point *is* a modulation vector
  over a shared implicit network; downstream tasks run on modulations. The
  "address is the datum" claim, already tested on images.
- **SIREN** (Sitzmann et al. 2020) and Fourier features (Tancik et al. 2020):
  coordinate networks cannot fit high-frequency signals without a harmonic
  basis. Directly relevant to why frozen phases are doing work here.
- **N-BEATS** (Oreshkin et al. 2020): the interpretable variant emits basis
  coefficients and the whole forecast horizon comes from evaluating them, no
  autoregression. Won M4. The existence proof for one-shot full-horizon - and
  also the clearest statement of its domain, since it works on low-entropy,
  near-unimodal signals.
- **Neural Processes** (Garnelo et al. 2018), **Latent ODE** (Rubanova, Chen,
  Duvenaud 2019): the probabilistic and continuous-time versions of one latent
  to a whole trajectory.
- **Analog Bits / Bit Diffusion** (Chen, Zhang, Hinton 2022) and
  **Diffusion-LM** (Li et al. 2022): the closest existing attempts to treat
  discrete text as a continuous object. Both work; both underperform
  autoregression at scale, for the multimodality reason below.

## Open questions

- **Multimodality.** Committing to $\tau$ far ahead means averaging
  incompatible futures - the failure that makes non-autoregressive translation
  need sequence-level distillation from an AR teacher. Speculative decoding
  sidesteps it entirely (a wrong draft is rejected, not emitted), which is why
  this is a *draft head* proposal and not a proposal to replace the trunk
  objective. Keep it that way.
- **Does the codec latent have the smoothness the bytes lack?** The whole bet.
  Text has no smoothness in position; the claim is that the continuous latent
  underneath does. The $\tau > K$ curve is the first evidence either way.
- **Does the field need to be queried at $t+\tau$, or at a learned coordinate?**
  Absolute position is the obvious first answer and matches how the field is
  already anchored. A learned warp of the query coordinate is the version that
  starts to look like an actual operator learner, and should not be attempted
  before the simple one is measured.
