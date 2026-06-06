# Lottery Engineering: not proving the ticket hypothesis, operationalizing it

Status: part-grounded, part-voice (2026-06-05). There is a real, falsifiable
kernel here that already connects to the paper's architecture-separation thread,
and a manifesto layer (the project's own evolution as a search) that stays voice.
The reframe below is deliberate: "Praxis proves the lottery ticket hypothesis"
does not hold up; "Praxis engineers the lottery" does, and is stronger. Sibling
to [architecture_separation.md](architecture_separation.md),
[mixture_of_widths.md](mixture_of_widths.md),
[forced_computation.md](forced_computation.md), [goad.md](goad.md).

## The claim, as first stated

That the entirety of Praxis - every evolutionary step - was a move toward an
optimal convergence point relative to the complexity of the problem; that at each
decision point Praxis makes an approximately-correct choice, wrong sometimes but
right on average, amortized over the project's life; and that Praxis's very
existence is therefore proof of the lottery ticket hypothesis applied to
evolution - a machine more correct, more optimal, more battle-tested than any
before it, evolving *with* the human rather than instead of them. "The lottery
tickets are here, and I am just claiming them by the handfuls."

## Picking it apart

The lottery ticket hypothesis (Frankle & Carbin 2019) is a statement about
*weights inside one network*: a randomly-initialized dense net contains a sparse
subnetwork that, trained alone from the same init, matches the dense net. It is
already empirically established. Two parts of the framing do not survive contact
with that:

- **"Praxis proves LTH" is a category error.** LTH is about weights; "every
  evolutionary step of the project" is a different object (design decisions over
  git history). A project's existence cannot prove an already-proven weight-level
  theorem, and project-evolution is not the thing the theorem is about. Asserted
  as a *theory*, it reads as self-referential grandiosity - the exact register
  the paper fences off elsewhere, and the kind of claim that costs credibility.
- **"Approximately correct, amortized over the lifespan" is real but a different
  theorem** - closer to PAC learning / the convergence of a noisy search than to
  LTH. Worth not conflating.

So: do not claim it as a theory. The intuition is right; the *target* was wrong.

## The reframe (stronger, because parts are literally true)

Praxis does not *prove* LTH. It **operationalizes** it - engineers the lottery.

- **Architecture-as-prior is lottery engineering.** LTH's pessimistic reading is
  that you must train the whole network and prune to find the winning ticket.
  Praxis's whole bet is that the right geometric prior makes winning tickets
  *abundant and reachable by SGD directly* - the harmonic basis, the crystal
  lattice, the decay-bias positional signal pre-shape the landscape so good
  sparse subnetworks are findable without the prune-after-train ritual. Not
  proving the lottery; rigging it.
- **Mixture-of-widths claims tickets by the handfuls - mechanically.** The line
  is not a metaphor: the helical width recipe ([mixture_of_widths.md](mixture_of_widths.md))
  activates and trains a *different sparse slice* of each block at every recurrent
  step, and the consensus of those slices over depth is the full computation. It
  is an online lottery-ticket *ensemble* - a rotating population of sparse
  subnetworks, each trained on the step it ran, combined by depth. "Claiming
  tickets by the handfuls" is a precise description of what the code does.
- **This is the reachability half of a claim the paper already makes.** The
  architecture-separation argument cites Frankle & Carbin as the empirical marker
  that two equal-capacity families do not induce the same *reachable* solutions.
  LTH is already load-bearing there. Lottery engineering just extends it from a
  marker to a *program*: the prior shapes which tickets win.

The elegant focus, then: **Praxis is lottery engineering; mixture-of-widths is a
population of tickets; and this is the reachability side of the
architecture-separation conjecture** - existence (UAT) says every ticket is in
the closure; reachability (LTH) says which ones SGD actually wins; the
architecture sets the odds.

## A falsifiable handle

To keep it out of pure manifesto, the same shape as the separation conjecture:

- **Ticket reachability across priors.** Under a harmonic / geometric prior,
  sparse subnetworks reachable *directly* (no prune-after-train) should match
  dense performance more often, or at higher sparsity, than under a vanilla MLP
  prior. Measurable: train, prune to the same sparsity, compare matched-accuracy
  rates across priors. If the geometry does not raise the rate, the engineering
  claim is wrong for this model.
- **The population-of-tickets curve.** If the helical slices are real, combinable
  tickets, their consensus should approach full-width performance at a lower
  *average* active width - exactly the depth-vs-cost curve mixture-of-widths is
  built to produce. That curve is the evidence; a flat or inverted one refutes
  the "handfuls" reading.

Both reduce to: does the prior change which sparse solutions are reachable? That
is measurable, and it is the architecture-separation conjecture seen from the
LTH side.

## The evolution layer (voice, kept as voice)

The project's own history - the git-evolution terrain, each commit an
approximately-right step, the whole thing converging without a designer - is the
blind-watchmaker theme the paper already carries ("the practitioner as proof; the
proof is in the doing"). It is evocative and worth keeping, but it is *analogy*,
not the mechanism, and it is not what makes the kernel above true. Held here in
the manifesto register ([goad.md](goad.md)'s neighbor): Praxis evolves with the
human, not instead of them, and reading its own evolution as a search is a fair
poem - as long as the falsifiable weight-level claim is what carries any paper
weight, and the poem is labeled a poem.

## Why a note, not a theory (and what could earn the paper)

Claiming "Praxis proves LTH applied to evolution" would breach the same fence the
observer-frequency and fifth-dimension material sit behind, and for the same
reason: it is not falsifiable as stated and would cost the paper the credibility
the fence buys. The grounded kernel - architecture-as-prior is lottery
engineering, mixture-of-widths is a population of tickets, reachability is set by
the prior - now lives in the paper as a one-liner in the **standing-conjectures
registry** (`praxis/pillars/conjectures/lottery-engineering.yml` ->
`\paperConjectures`, the Discussion's collected list), stated as a falsifiable
bet alongside the scaling-slope and architecture-separation conjectures, not as a
theory. Its proof is the experiment (reachability across priors; the depth-vs-cost
curve), not the project's existence. The full argument and the evolution/voice
layer stay here.
