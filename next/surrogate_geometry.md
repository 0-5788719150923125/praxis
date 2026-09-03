# Three proposals from a Discord thread, resolved

Reviewed 2026-09-02 (`PROPOSAL.md`): surrogate pixel-grid modality,
precision-gated harmonic deltas, and `const [t, t]` as a fixed attention
geometry.

**Two are closed. The third became [kaleidoscope.md](kaleidoscope.md) and is
built.** This note is kept short and only so the closed two do not arrive a
third time - they had already been raised once and parked in
[integration_backlog.md](integration_backlog.md).

## Closed: surrogate representations via pixel grids

Item 6 of [integration_backlog.md](integration_backlog.md), already settled in
[magnetism.md](magnetism.md). The proposal states its own killer and walks past
it: it works "only if the surrogate is genuinely orthogonal to the token stream,
not just a rendering of it," and a pixel grid of text at a fixed font is exactly
a rendering. `H(pixels | text) = 0`, so there is no independent view and no
nuisance variable to be invariant to. What pixels genuinely deliver -
sub-token orthographic structure with no vocabulary bottleneck - is already
bought by `byte_level` + `byte_multihash`.

It is also CALM work, and CALM was shelved 2026-07-16 by deliberate decision
([[project_calm_direction]]). The CALM conditional it proposes to fix was
root-caused (`roadmap.md:134`): the energy head learns the marginal, not the
conditional, and an auxiliary anchor moved next-patch 0.003 -> 0.139.

**The one surviving fragment**, if anyone wants it: for a byte-level model,
"render text as pixels" is mechanically a frozen `[256, H*W]` glyph atlas at
21 KiB, looked up with `F.embedding` - not vision, an embedding prior, no new
dependency. Confusability structure in that basis is real (`'O'`/`'0'` cosine
0.973, `'E'`/`'F'` 0.890). Enters through `EMBEDDING_REGISTRY` beside
`byte_multihash`. Honest prior: null, since it adds no information, only a bias
about which bytes start near each other.

## Closed: precision and harmonic deltas

The premise is false for the runs cited. `abstractinator-a.yml:123` sets
`precision: float32`, every `abstractinator-{b..h}` extends it, and
`DEFAULT_PRECISION` is `float32` regardless. There is no bfloat16 in this
lineage to be quantizing a delta away. The conditional path is fp32 internally
in any case: `_field_conditional` opens with `hidden_states.float()` and only
casts back at the end.

**What it did surface, and this part shipped:** the claim it rests on ("99% of
computation lives in the static spectrum, 1% in the chirp") was not readable
from anything we logged. `capacity_split` divides by `bias + variance +
dormant`, and dormant dominates a concentrated field, so both shares are driven
toward zero and "1%" was consistent with the delta carrying anywhere from
nothing to a third of the written field. `harmonic_variance_share` =
`variance / (bias + variance)` now reports the actual split, 0 at init by
construction. **Read it on the next run** - if it sits at 0, the
input-conditional envelope is inert, which is a bigger finding than any of the
three proposals.

## Three metric misreadings, worth remembering

The proposal justified itself with four observations. Three do not say what they
were read to say, and all three are easy to repeat:

1. **`chirp` is a `Servant` activation metric**, the token-dispersion of an FFN
   frequency swing. It is not the harmonic head's input-conditional delta. Two
   unconnected modules.
2. **`val_bits_per_byte` is `val_loss / log(2)`**, and `loss_func: halo` is a
   composite of CE plus a geometric term. Its own description says the level is
   not calibrated under a composite objective and names `val_byte_nll_bits` as
   the calibrated companion, where chance is 8.0.
3. **`rlct_manifold_var` projects one weight tensor**, not the model, and its
   own description reads high top-2 variance as "nearly planar, a flat
   **anisotropic** sheet" - which is what `contrastive_isotropy` exists to
   fight. A rising number there is not self-evidently good.

## Open: `const [t, t]`

Built as Kaleidoscope. See [kaleidoscope.md](kaleidoscope.md).
