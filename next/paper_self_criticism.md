# Paper self-criticism: the register of soft claims

> Status: **open, standing** (opened 2026-08-23). Not a thread and not a
> feature - a working list of claims in `research/` that read as assertions
> but are not yet falsifiable, each paired with the observation that would
> harden or kill it. §5.10 *Limitations* in `body.tex` is the published face
> of this; that section names what is unestablished. This note is the
> workbench behind it: it also holds the claims that have not been demoted to
> Limitations yet, and, for each, the specific measurement that settles it.

## The test

A claim is soft when the answer to **"which architecture would fail this
description?"** is *none*. A statement true of every model distinguishes no
model, however well it reads. The paper already applies this test correctly
once, to the Koopman claim (`body.tex:44`):

> The claim has a boundary, and is therefore falsifiable: a representation
> whose basis depends on the input, that is not a superposition of a fixed set
> of modes, or whose dynamics are not approximately linear in any fixed basis,
> is *not* harmonic here.

That is the standard. Every entry below is measured against it.

## The fix pattern

1. **State the boundary** - what would *not* count as an instance.
2. **Name the observation** - an existing logged metric wherever possible, not
   a study that would have to be commissioned.
3. **Name the falsifier** - the outcome that retires the claim, in advance.
4. **Put it in a framing fragment**, not in `body.tex`. Even run-independent
   prose belongs in an always-active fragment: the body is meant to shrink,
   and a fragment carries its own `\cite` with it. See
   `feedback_paper_conditional_references` in memory.

## Entries

### 1. "Underneath the tokens this is time-series modeling" - HARDENED 2026-08-23

**Was** (`body.tex:197`): "underneath the tokens the model is not doing
language modeling but *time-series modeling over continuous samples*."

**Why it was soft:** any architecture that embeds discrete symbols in
$\mathbb{R}^d$ can be narrated as fitting a real-valued signal. The reframe
described a genre, not a mechanism, and would have been read as one.

**What changed:** the absolutism came out of the body, and a three-fragment
family now carries the boundary and the two places where the fitted-signal
reading and the token reading actually disagree:

- `framing/extrapolation-claim.yml` (always active) - the boundary, plus the
  two discriminators: **where the error lives** (event-driven sparsity,
  already tested by `harmonic_delta_norm` concentration) and **off the sampled
  interval** (a fitted signal has values where no sample was taken; a token
  sequence does not). Falsifier stated: a read that is accurate inside the
  consumed span and collapses at its edge is a memorized window, and the
  address language should be dropped.
- `framing/extrapolation-mtp.yml` (any `mtp_type`) - MTP *is* the
  off-interval query; `mtp_draft_acc_d{k}` is the decay curve. Two honesty
  guards: any AR state carries some information about later positions, so the
  *shape* is the claim and not the height; and the head trains on all $K$
  offsets, so the profile measures interpolation inside the trained window.
- `framing/extrapolation-serpent-rnn.yml` (`mtp_type: serpent_rnn`) - the
  clean test is a read at $k > K$, and the shared cell is the one bank where
  it is nearly free.

New inline `paperMtpAcceptRun` (`mtp_accept_run`) reports the *realized*
horizon once generation has run, and stays silent before that.

**Residual work - the experiment that closes this entry:**

The $k > K$ unroll. `praxis/heads/mtp/rnn.py` is one gated cell
unrolled $K$ times; the only per-depth parameter is a zero-initialized
signature. Continue the unroll to $k = 2K$ holding the signature at zero,
score against the true bytes, and plot accuracy against $k$. Two outcomes,
both publishable:

- **graded decay across the trained boundary** - the state is an address into
  a basis, the fitted-signal reading survives, and the paper can say so with a
  curve instead of an assertion;
- **a cliff at $k = K$** - the depths memorized their supervised offsets, the
  "address into a fixed continuous space" language is decoration, and entry 1
  moves to Limitations instead.

Cheap: no training, an eval hook plus a chart. The architecture that would
make this query native rather than improvised is designed in
[field_query.md](field_query.md), which is gated on this measurement coming
back graded rather than cliffed. Do it before the next paper
build that claims the reading. Until then the fragments say only what has been
measured - the profile "stops where the training does" - which is why they can
ship now.

### 2. Candidates not yet audited

Not yet worked, listed so they are not rediscovered from scratch:

- **The Library of Babel framing** (`body.tex:195`). Decorative on its
  surface but it does assert something checkable: that reconstruct-consumed
  and predict-next are "two neighboring pages read off by the same invertible
  transform." Invertibility of the linear solve is a property of the codec
  that can be measured (round-trip error), not asserted. Either measure it or
  drop "invertible."
- **"The outer representation"** (`body.tex:185`). The claim is that learning
  pressure falls at the rim rather than the interior. That is a claim about a
  *distribution over positions*, so it wants a per-position loss profile -
  and `outer-objectives-standard.yml` already correctly frames the
  no-codec run as the control for it. The measurement is the missing half.
- **"Orthogonal axes"** for the bias-variance decoupling. Already demoted in
  §5.10 as "an interpretation of those diagnostics, not a theorem" - correct,
  and it stays there until someone measures the correlation between the two
  diagnostics across runs. If they co-vary, "orthogonal" is wrong.

## Rules of engagement

- An entry is *closed* only by a measurement, never by a rewrite. Rewriting
  moves a claim from "asserted" to "falsifiable"; that is a different, lesser
  achievement, and entry 1 records both states separately for that reason.
- A claim the current run cannot test does not go in the paper unconditionally.
  Gate it, or leave it here.
- Prefer an already-logged metric to a proposed one. Nearly every entry above
  is settled by something the dashboard already records.
