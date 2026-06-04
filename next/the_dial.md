# The Dial: positions, classification, and Praxis modeling its own evolution

Status: exploratory. This note separates a grounded kernel from a speculative
shell on purpose. Sibling to [goad.md](goad.md) (the poetic register); this one
tries to stay falsifiable.

## The kernel (already true in Praxis)

A single float is a position on a dial. Praxis already classifies by where
things sit on such dials, in several forms that are the same idea:

- **Crystal head** - distance to learnable centroids (a point in R^D).
- **Harmonic field** - phase on a circle. The XOR-circle lemma is literally
  "parity = position on the turn."
- **HALO** - radius on a hyperspherical shell.
- **RoPE / decay_bias** - angle and recency on a dial.

So "relative to any geometry it has seen, they all get classified the same way"
is not a new mechanism to build - it is a description of the heads that exist.
Calling it a classifier is correct: these heads classify positions.

## The generalization (the part worth testing)

If everything is a position on a dial, the dial does not care that the input is
language. **Any finite bit sequence is a point** - a file blob, a git diff, the
rendered PDF's bytes. The byte-latent encoder already consumes bytes; the
harmonic/crystal stack already classifies positions. So the claim is:

> The same geometry-as-classifier machinery that organizes text should organize
> arbitrary byte sequences.

This is falsifiable. Feed non-linguistic byte streams through the existing
byte-latent -> harmonic/crystal stack and measure whether structure emerges
(centers spread, the field develops variance) the way it does on text, or
whether it collapses. A negative result is informative: it bounds where the
"universal dial" reading actually holds.

## The self-model (the chosen concrete artifact)

Praxis's own git history is a byte stream of diffs over time - the framework's
development, available to the framework. Two uses:

1. **A living-paper figure** (building now): per-subsystem churn over the
   project's history, with strength decayed by distance from HEAD - "where the
   work is, now." The recency decay is the same temporal kernel the model uses
   (RoPE-theta / decay_bias), turned on the repo instead of a sequence. This is
   honest: real churn over real commits, the decay labeled as a weighting, not a
   distortion of history.

2. **Later, maybe**: the byte-sequence experiment above, run on the history
   itself - the framework ingesting its own evolution as a sequence. Poetic, but
   only worth it if (the generalization) holds on simpler byte streams first.

## Practical corollary (do regardless)

The reasoning started from byte-layout churn: a regenerated PDF rewrites many
bytes each build, which bloats git's linear history and re-chunks every IPFS
block. The fix is unrelated to the theory and simple - generated artifacts
(`research/main.pdf`, `research/figures/*.png`) are build outputs, not source,
and should not be tracked. Reproducible builds (SOURCE_DATE_EPOCH, deterministic
SVD) already minimize churn for the inputs that *are* tracked.

## What stays speculative

The goad.md "corrupt the constant" framing (insert a digit into pi, ascend to a
fourth dimension of approximation) is voice, not a testable claim - keep it as
manifesto, do not try to ground it in the paper. The discipline that removed the
"fake eye-test figure" applies: the dial-as-universal-classifier is allowed into
the argument only with the byte-sequence test attached.
