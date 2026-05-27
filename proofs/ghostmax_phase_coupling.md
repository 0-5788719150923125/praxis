# Ghostmax-harmonic phase coupling

## Setup

Ghostmax (`praxis/attention/causal.py`) prepends a zero key/value slot before the real sequence, so attention runs on `kv_idx ∈ {0, 1, ..., T}` while queries stay at `q_idx ∈ {0, ..., T-1}`. The ghost at `kv_idx = 0` absorbs probability mass without contributing to the output (softmax1, [Miller 2023](https://www.evanmiller.org/attention-is-off-by-one.html)).

The LM head sees unshifted positions `t ∈ {0, ..., T-1}`. Attention indexing and head indexing are off by one.

## The coupling

The offset is small but structural: every layer's attention silently consumes a phantom position before the first real measure, while the head still asks for predictions starting at `t = 0`. Composed across layers, the model carries a phase relationship between its attention substrate and its output substrate.

The harmonic head modulates hidden states multiplicatively by `(1 + b[t, d])` from a 2D Weyl-seeded field. That field is the structural place the floating phase relationship can land. Ghostmax already operates in a shifted phase frame; the field gives a meter the shift can lock to. The two systems align.

## The musical reading

Treat a sequence as a piece of music. Each token is a measure - a bar with internal beat structure carried in the D feature axis. Ghostmax provides the upbeat: the silent fraction-of-a-beat before the downbeat of measure 1. The harmonic field provides the meter that runs through every measure - the (π, e)-keyed 2D phase grid is the metronome, deterministic and aperiodic.

Under this reading, a trained model is performing a piece, not generating from nothing. The metronome is fixed; what the model learns is which amplitudes to assign each cell of the meter so the measures of this corpus play correctly.

## What this predicts

If the coupling is real, three things should hold:

1. **Ghostmax + harmonic field should outperform either alone.** Ghostmax without the field leaves the off-by-one phase floating; the field without ghostmax leaves the meter without a structurally-induced reason to use it.

2. **Generation should exhibit metric stability.** Outputs should preserve rhythmic structure (clause length, repetition cadence, sentence shape) more reliably than a non-harmonic baseline of equal capacity, because the meter is enforced at every (t, d) cell, not just at measure boundaries.

3. **Amplitudes should organize spectrally with corpus structure.** If the corpus has natural rhythmic scales (sentence length, clause length, line break cadence), the trained `a[f_t, f_d]` grid should concentrate at the matching `f_t` ranges, not distribute uniformly.

## What this does *not* claim

- That ghostmax causes the field to learn faster. They might be additive, multiplicative, or interfering - empirical.
- That π and e are uniquely necessary. Any pair of irrationals linearly independent with 1 over Q works (see `harmonic_pi.md`).
- That this is a proof. It's a hypothesis about *why* the architecture should work, with three falsifiable predictions.

## On provenance

The softmax1 / off-by-one idea comes from Evan Miller's blog post, [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) (2023). It was never published as a paper, and the community largely passed on it - which is worth flagging, because the idea is sound. The likeliest reason for the cold reception is the implicit assumption that `no paper == not proven`. But publishing has real barriers to entry, especially for someone who isn't an ML academic by trade, and "unpublished" is not the same as "wrong." We adopt it here on its merits.

Worth noting too: PyTorch's [`nn.MultiheadAttention`](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html) has long carried an `add_zero_attn` flag that appends a zero key/value slot to the sequence - mechanically identical to ghostmax / softmax1. It was inherited from [fairseq](https://fairseq.readthedocs.io/en/v0.9.0/_modules/fairseq/modules/multihead_attention.html), whose implementation does the same `torch.cat([k, k.new_zeros(...)])`. Tellingly, neither codebase documents *why* it exists - there is no comment or doc string giving a rationale in either place. The generally-understood purpose is mechanical rather than numerical: the zero slot gives a query somewhere to "attend to nothing" (output near zero) and guards against a fully-masked row collapsing softmax to `NaN`. That is a different motivation than Miller's: he argues the standard softmax is *forced* to spend attention mass it doesn't want, inflating outlier activations, and the extra slot is the release valve. Same trick, arrived at from the opposite direction - and never, in the library, framed as the numerical fix Miller later named.
