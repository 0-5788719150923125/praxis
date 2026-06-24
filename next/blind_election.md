# Blind Election: Issue-Consensus as a Voting Primitive

> Status: **flagged voice, heaviest-caveat register (2026-06-23).** A mechanism-
> design / social-choice thought experiment, deliberately kept _separate_ from
> [temperament_bias_variance.md](temperament_bias_variance.md): that note is a
> structural lens that is silent on policy _by construction_, and the moment you
> prescribe how voting should work you forfeit that silence. This note is the
> prescriptive sibling, so it carries the prescription's burden of proof. It is
> the _one politics-adjacent mechanism_ thread, engaged only because it maps onto
> voting primitives the repo already builds. Sibling in tone to the antimatter
> polarity and the 144,000 - use to think, not to advocate.

## The one line

**Decompose the vote into issues, derive the executive from consensus across
them, and you've rebuilt the CALM patch-vote as a polity - inheriting both its
appeal and its known failure modes (the discursive dilemma, and consistency-
weighting as entrenchment).** Everything below is bound by that sentence.

## The proposal (as posed)

Citizens vote on _issues_, not on a person. The president (the executive verdict)
is then _derived_ from consensus across those issue-votes, rather than chosen
directly. Optionally, weight a voter's influence by the _consistency_ of their
history - reward coherent, stable preference profiles over erratic ones. "Blind"
because the choice of person is decoupled from, and downstream of, the issues - a
veil of ignorance between the votes and the office.

## Why it belongs near this repo at all

It is not a new political idea so much as a familiar _Praxis_ one wearing a civic
costume. Three near-exact mappings to machinery already in the tree:

- **Issue-decomposed consensus = the CALM patch-vote, generalized.** Praxis
  already aggregates per-dimension distributions and reads off a winner rather than
  voting on the winner directly (`_patch_vote_sample`, the count-based vote). "Vote
  on issues, derive the executive" is that move at polity scale.
- **Consistency-weighting = reliability-weighted / epistemic voting.** The
  elimination-tournament item and the swarm's stratified vote already upweight
  competent or stable participants - the Condorcet-jury-theorem intuition that a
  better-calibrated voter should count more.
- **"Blind" = the paper's decoupling move.** Separating the person from the issues
  is the same refusal-to-collapse-axes that
  [temperament_bias_variance.md](temperament_bias_variance.md) makes with bias and
  variance: don't fuse two orthogonal quantities into one partisan dial.

The name rhymes with Blind Watchmaker on purpose; the kinship is the aggregation-
of-many-small-decisions-into-one-emergent-outcome shape.

## The two hard problems (lead with these, they are the point)

### 1. The discursive dilemma (doctrinal paradox)

This is the load-bearing risk and the reason the proposal is non-trivial.
**Aggregating issue-by-issue and aggregating the verdict can disagree.** A
majority can endorse each _premise_ while a majority simultaneously rejects the
_conclusion_ that those premises entail - so "consensus across issues" and
"consensus on the resulting executive" are not the same object, and can point at
different people. (List & Pettit, judgment-aggregation theory; the doctrinal
paradox from jurisprudence.) Any real Blind Election must _choose_ a side of this -
premise-based vs conclusion-based aggregation - as a deliberate design commitment,
not stumble into it. There is no neutral default; the choice is the design.

### 2. Consistency-weighting is the monoculture failure mode, by construction

The sharpest internal tension, and the one that keeps this honest: weighting by
_consistent histories_ is **high-bias by definition**, and the temperament note's
own skeptical spine - Wald's bombers, the cost of unopposed consensus - is a direct
indictment of it. Upweighting the consistent entrenches the prior and quietly
silences the dissenters who already left the dataset; "consistency" rewards the
survivors and mistakes them for the whole truth. So the consistency knob is not a
free virtue - it is a **variance-suppressor**, and an unbounded one ossifies the
polity into exactly the echo chamber the corpus warns against. If this mechanism is
ever defensible, the consistency-weight has to be _bounded_ (and arguably paired
with a deliberate variance term - structured dissent that is upweighted, not down),
so the system rides the decoupled interior rather than collapsing onto pure bias.

## Continuous suffrage: the continuum that points the wrong way

The deepest and most dangerous extension, and the one most native to the corpus's
instincts - which is exactly why it needs the heaviest guard. Suffrage today is a
**binary gate**: enfranchised or not, a step function on the influence axis, the
most discrete object imaginable. The natural Praxis move is to do to it what the
paper does to bias/variance and what the rank-priced connection does to bandwidth:
**replace the step with a continuum.** Every citizen's influence becomes a weight
in `[0,1]` (or continuous positive reals); nobody fully enfranchised, nobody fully
zeroed; all voters partial, all the time. The discrete edge dissolves and each
voter becomes a _point - a little geometry - in influence space_, all of them
coexisting and superposing rather than sorted into two bins. **Fractional participation
is already a Praxis primitive; voting weight _is_ rank.**

**The turn (do not flinch from it).** Everywhere else in the corpus, continuous-
over-discrete is the _good_ direction - the decoupled interior, the harmonic field,
the smooth manifold. **Here the discreteness is the protection and the continuity
is the attack surface.** One-person-one-vote is a hard-won Schelling point
_because_ it is binary and equal: a weight that cannot be tuned is a weight nobody
can quietly tune _down_. Every historical instance of weighted or partial suffrage

- poll taxes, literacy tests, property qualifications, the three-fifths clause, the
  whole Jim Crow apparatus - was precisely a continuous re-weighting of influence,
  always laundered as neutral. "Remove partial voting rights from ALL citizens,
  symmetrically" _sounds_ egalitarian, but a continuous weight is a strictly more
  powerful instrument of entrenchment than a binary gate: deniable, infinitely
  tunable, and **whoever sets the weighting function wins** while a smooth knob hides
  its own thumb on the scale. The symmetry is illusory - weights are never applied
  symmetrically in practice; that asymmetry is what makes them weights.

**The finding, not a bug:** the corpus's continuous-geometry instinct and
democratic equality point in _opposite_ directions here. Continuity is the richer
object and the more dangerous one; the binary floor of equal suffrage is a
_deliberate refusal of the geometry_ - a constitutional guarantee that no human's
influence can be driven toward zero by any function, however elegant. This is the
consistency-weighting caveat at its limit: the variance/dissent floor is no longer
a tunable term but the **irreducible discrete `1`.** The honest lesson cuts back
into the ML work too - it is a standing counterexample to "smooth is always better,"
and a reminder that some discrete equalities are load-bearing precisely because
they cannot be re-weighted.

## The hard caveats (co-equal, not a footnote)

- This is a **thought experiment in mechanism design**, not advocacy for any real
  electoral reform, party, movement, or person.
- It says **nothing about who should win** any concrete question - it is a claim
  about the _structure of aggregation_, and aggregation structure is famously able
  to manufacture or reverse outcomes (Arrow, Gibbard-Satterthwaite: no aggregation
  rule is simultaneously fair, non-dictatorial, and strategy-proof in general).
- Real elections are adversarial: any derived-executive rule invites **strategic
  issue-voting** (vote your premises insincerely to swing the conclusion), which the
  count-based vote does not have to survive but a polity does.
- This is among the **most speculative and most easily-abused mappings in the
  corpus**, same flagged-voice register as the temperament note. Use it to reason
  about _consensus-aggregation tradeoffs_, nothing more.

## Prior-art anchors (contested; NOT workflow-verified)

Judgment aggregation / discursive dilemma (List & Pettit); Condorcet jury theorem
and epistemic democracy (the case _for_ competence-weighting, and its critics);
Arrow's impossibility and Gibbard-Satterthwaite (the structural limits any such
rule hits); liquid / delegative democracy and quadratic voting (modern mechanism-
design cousins); Rawls' veil of ignorance (the "blind" framing). None of these
endorse a Blind Election - they are the literature against which a careful version
would have to argue, and several of which (Arrow, the doctrinal paradox) it must
actively survive rather than ignore. Verify before any of this hardens into a
claim.

## Grounded shadows (where it touches real mechanism)

- The aggregation core is literally the **CALM patch-vote** (`_patch_vote_sample`,
  count-based vote) and the swarm's stratified vote - issue-consensus is those, at
  scale.
- Consistency-weighting is the **elimination-tournament / reliability-weighted**
  vote, and inherits its open question (consistency vs confidence vs held-out
  agreement as the weight).
- The bounded-consistency-plus-dissent corrective is the paper's **decoupled
  interior** ([temperament_bias_variance.md], [observer_grounding.md] 95/5): keep a
  low-bias variance channel load-bearing rather than letting consistency dominate.
- The pool-of-candidates-collapsed-by-similarity framing echoes the **pool-HEAD**
  roadmap item - many near-consensus candidates tolerated, then resolved.
