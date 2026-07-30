# The Motor Channel: One Continuous Action Space, Text Included

> Status: **speculative / unscoped** (2026-07-30). The idea: give an agent exactly
> one output - a point and a click - and make text emission a special case of
> pointing rather than a privileged channel with its own head. The on-screen
> keyboard is the mechanism; the layout of that keyboard is a free variable and
> the actual research question. Companion to [world_models.md](world_models.md)
> and [mtp_curve.md](mtp_curve.md).

## The thesis, stated plainly

A computer-use agent normally has two mouths. It emits tokens (its native act)
and it emits GUI actions (clicks, drags, keypresses), and something in the middle
has to arbitrate between them - tool-call syntax, a router, a mode flag. Those two
channels have different geometries, different losses, and different failure modes,
and the seam between them is where most agent scaffolding lives.

The proposal removes one mouth. The agent's **only** output is a continuous
position and a commit signal. Typing, pressing a button, dragging a window,
drawing a curve, moving a slider - all the same action, differing only in where
the pointer went and what happened to be rendered there. To write text, the agent
opens an on-screen keyboard and points at it.

Text stops being a modality and becomes a place you go.

That is the claim worth testing. Not efficiency (this will never be the fast
path), and not the "regions are tolerant of error" argument that motivated the
first sketch - see below, that argument does not hold. The claim is **modality
collapse**: one head, one loss, one geometry, no arbitration seam.

## What the radial layout does and does not buy

The initial sketch: a circle sliced like a pizza, one slice per character, and the
model predicts (angle, radius). The stated appeal was tolerance - direction and
distance become *regions*, so small errors are absorbed.

That argument does not survive contact. A softmax over 128 characters is already
maximally tolerant: any logit ordering that puts the right character on top wins,
with no geometry to get wrong. Converting to polar coordinates and discretizing by
region does not add slack; it adds a quantization step and puts hard
discontinuities at every slice boundary. Worse, if angular adjacency is arbitrary
(slice 7 is `q`, slice 8 is `K`), a small angular error is a catastrophic semantic
error and the regression loss cannot tell the difference. Regression onto a target
whose semantics are categorical is a known way to get a bad surface.

So the geometry has to earn its place some other way. Three ways it can:

**1. Order the slices so angular distance means something.** Lay characters out by
embedding similarity, or by a learned 1D ordering, so that a near-miss in angle is
a near-miss in character. Then angular error approximates substitution cost, and
the gradient on the angle is informative rather than adversarial. This is the
single change that turns the geometry from a liability into a signal.

**2. Make the second dimension carry real information.** In the flat pizza,
radius is decorative. Two ways to fix that, and they conflict, so it is a real
choice:

- *Nested slices.* A slice holds a **group** of characters (an embedding cluster),
  angle picks the group, radius picks the member. A two-level hierarchical decode,
  and it scales - a 16-slice wheel with 8 rings addresses 128 characters with two
  coarse continuous decisions instead of one fine one.
- *Radius as confidence.* Gesture magnitude tracks certainty; short travel means
  "not committed yet." A calibration readout for free, with no extra head.

**3. The vortex, which is the most interesting version.** A spiral running outward
from the center, characters placed **along the curve in probability order** -
most likely nearest the center. The action collapses back to a single arc-length
coordinate embedded in 2D, and distance now means *rank in the prior*. Short
movement = high-probability character. The elegant consequence: expected gesture
magnitude equals the entropy of the prior. The interface measures its own
uncertainty as literal distance travelled. That is Dasher's information-theoretic
property recovered in a form the model produces rather than consumes.

Note the vortex spends the radius channel on rank, so it cannot also spend it on
confidence. Pick one, or nest a vortex inside each slice of a coarse wheel.

## Prior art you have to read before building

**Dasher** (Ward & MacKay, Cambridge Inference Group, ~2000-2002) is the closest
thing to the adaptive-keyboard half of this. A zooming text-entry interface driven
by continuous 1-2D pointing, where a language model sizes the target regions in
proportion to probability. Built for eye-tracking and other low-bandwidth input,
measured in real words-per-minute, information-theoretically motivated.

Two things to take from it. First, the mechanism demonstrably works as a text
channel - this is not a guess. Second, and this is the caution: Dasher's
efficiency argument depends on the **pointing channel being the bottleneck**. You
get `log2(1/p)` bits per gesture because the regions are sized by probability, and
that is worth something only when the pointer is a noisy human eye and the prior
lives elsewhere. If the model shaping the keyboard *is* the model pointing at it,
the keyboard is a readout of the agent's own belief and clicking it is a
geometric detour around a sample it already had. **The interface only earns its
keep when the prior is external to the agent, or the actuator is genuinely
noisy.** Decide which of those is true before building, because it determines
whether the whole thing is circular.

**Swipe keyboards** (Swype, SwiftKey Flow) are the other reference: one continuous
trajectory per *word*, decoded against an LM. Far better bandwidth story than one
click per character, and a direct answer to the cost problem below.

**Computer-use agents** are table stakes and not the novelty. The novelty claim
rests entirely on the single-channel framing.

## Two constraints that are not negotiable

**Shrink, never remove.** The sketch proposed pruning low-probability characters
off the keyboard ("X never follows Z"). Dasher never zeroes a region, and the
reason is a hard capability ceiling: hashes, base64, identifiers, non-English
text, adversarial strings, and every real "Zx" become physically unreachable. Area
proportional to probability, **floored**, never zero. A prior-shaped interface must
stay complete or the agent inherits the prior's blind spots as hard limits.

**The model must actually read the keyboard.** If the layout is known a priori,
the model learns "predict next character, apply fixed geometric transform" and the
geometry is decoration on a softmax. The rendered keyboard has to be parsed from
the screen every step for the mechanism to be doing any work. That is also the
source of the cost.

## The isomorphism to work already here

If the keyboard prior is a **separate, frozen** model and the agent steers it,
this is speculative decoding with a spatial interface: the keyboard is the draft
model, the pointer is the verifier, and every gesture is an accept/reject. That is
structurally the machinery in [mtp_curve.md](mtp_curve.md) and the MTP draft-width
work, rendered as geometry instead of a token loop. It is also the only
configuration where the loop is not circular (see the Dasher caution above), which
makes it the default design rather than a variant.

Worth noting the layout question then inherits MTP's adaptive-width result: the
keyboard should get *coarser* when the prior is confident and *finer* when it is
not, the same way draft width tracks the accepted-run EMA.

## The cost, honestly

Reading the keyboard off the screen means a vision encode plus a decode per
character. Roughly three to four orders of magnitude over token emission.
Per-word gestures (the swipe model) buy back maybe one. Nested slices buy a
constant factor. This will never be competitive as a text channel, and any
framing that leans on efficiency is a losing one. The case rests on the unified
action space, full stop.

## Layout is a registry, not a constant

The pizza is the first entry, not the design. Vortex, nested slices,
probability-ordered spiral, QWERTY-on-a-plane (as a control), coarse-wheel-of-
vortices - these are ablatable entries in a `LAYOUT_REGISTRY`, each a function
from (prior over characters) to (rendered geometry, decode function). Constants
live in the layout profile, not in new config scalars. This is the shape that
makes "we have to build it to find out" tractable rather than open-ended: the
research question becomes *which layout*, measured, rather than *does geometry
help*, argued.

## The cheapest first probe

No agent, no desktop, no computer-use stack. A closed environment:

1. Render a layout to a small image from a character prior. One `LAYOUT_REGISTRY`
   entry (start with pizza, embedding-ordered - the ordering fix is what makes the
   baseline honest).
2. A small vision encoder plus a two-output regression head predicts (θ, r).
   Supervise by imitation against the slice centroid for the known next character.
3. Measure **character error rate as a function of angular noise injected into the
   action**. This is the whole experiment. If the geometry buys nothing, CER rises
   as fast as a softmax's accuracy falls under equivalent logit noise, and the
   tolerance argument is dead on measurement rather than on argument.
4. Then swap the layout entry and re-run. Vortex vs pizza vs nested, same harness.

Two questions fall out of that, both answerable on one machine with no agent
infrastructure: does slice ordering by embedding actually make angular error
graceful, and does any layout beat softmax-under-noise at equal action-space
dimensionality. If both are null, the motor-channel thesis can still be true -
it just has to be argued on modality collapse alone, and the keyboard becomes an
implementation detail rather than the idea.

## Honest unknowns

- **Is the unified action space actually easier to learn, or just tidier to
  describe?** One head with one loss is aesthetically clean; nothing guarantees a
  motor policy learns text faster than a text policy learns clicking. The
  arbitration seam we are removing may be cheap to cross.
- **What supervises the gesture?** Imitation is trivially available (render the
  keyboard, take the target centroid) but teaches the fixed transform, not the
  reading. RL over task completion supervises the right thing and is far more
  expensive. The honest answer is probably imitation to bootstrap, RL to make the
  reading real, and that is a big lift.
- **Whose prior shapes the keyboard.** Answered above in principle (external and
  frozen, or the loop is circular) but unresolved in practice - a frozen prior that
  disagrees with the agent will fight it, and the accept/reject dynamics of that
  fight are exactly the MTP acceptance-rate question in a new coordinate system.
- **Does angular error actually correlate with semantic error** under an embedding
  ordering, or does a 1D projection of a high-dimensional character space destroy
  too much for the ordering to mean anything? This is what probe step 3 measures,
  and the first thing that can kill the geometry.
