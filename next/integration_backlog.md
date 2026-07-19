# Integration backlog - the 2026-07-18 idea dump

> Status: **parked by decision** (2026-07-18) - "I don't know what to do with
> any of this, so I don't want to do this work right now." This note replaces
> the deleted `PROMPT.md`. Two items graduated to their own notes the same
> day: the MTP curve ([mtp_curve.md](mtp_curve.md), now in the paper) and the
> multi-scale banks ([harmonic_fingerprint.md](harmonic_fingerprint.md)).
> Everything else is preserved here with a sober read on each.

## Observations of existing runs (framings, not new work)

1. **Geometric specialization via the VEAR router.** Under MTP (K=4 bytes
   per two passes) the experts learn complementary manifolds, not task
   divisions: expert 0 linear one-dimensional commitment, expert 1 radial 2D
   exploration, expert 2 an alternative axis, expert 3 diffuse hedging. The
   router learns which expert's *geometry* each token needs -
   manifold-matching, not task routing. Between-expert variance is
   consistent with (not proof of) specialization. Already partly captured in
   `experiments/abstractinator-a.yml`'s VEAR comment and the
   `vear_repulsion` cards.

2. **Harmonic constraint as regularization.** Weight decay disabled (except
   2D layers) because the harmonic basis already constrains the search.
   "Complete permanent overfitting" within a constrained geometry =
   commitment, not failure; the 5% freedom inside 95% shared geometry is
   where learning happens. Already in the paper as the MuonGeo weight-decay
   paragraph (`research/body.tex`) and the `experiments/abstractinator-b.yml`
   optimizer comment.

3. **Soft projection through bottlenecks.** Special tokens force state
   through narrow channels - not hard resets, rotations through
   lower-dimensional subspaces. The moire pattern shows which information
   survives, rotated into a new basis. Unformalized; would belong beside the
   bottleneck discussion if it ever firms up.

4. **Memory regime competition (NEAT + Titans).** Two incompatible learning
   regimes (energy/exponential vs. EML/log-minus-exponent) coexist under a
   both-must-survive constraint; blend width = allocation, brightness =
   forecast fitness. Already reported via the `memory-dual-regime` framing
   fragment.

5. **Convergence as harmonic settling.** The router converging at step 13773
   is a real observation; the claim that the step number "containing 1337"
   is meaningful is numerology and should not enter the paper. The testable
   remnant: do convergence events land at predicted harmonic multiples? No
   evidence gathered; treat as unfounded until someone bothers.

## Proposals (untested, would be new work)

6. **Surrogate representation via pixel grids.** CALM's unlearnable
   conditional as symbol collapse; break it with an orthogonal modality -
   pixel grids of the text, convolved, carrying spatial signal the token
   stream doesn't encode. Constraint: the grid must contain signal
   inaccessible to tokens, or it's noise. Note: this is CALM work, and the
   CALM direction was shelved 2026-07-16 - reopening it should be a
   deliberate choice, not a drift.

7. **Input-seeded initialization via [t, t] correlation.** Compute
   `C = input @ input.T`, extract principal modes (SVD), initialize weights
   as a learned blend of those modes - the data marks where gradients want
   to flow before training begins. Cheapest of the three to falsify:
   drop-in init change, clear metric (convergence speed vs. baseline). Open
   design question that changes its category: seeded once per model (from
   what data?) or per-sequence (then it's architecture, not init).

8. **Multi-scale memory banks via harmonic dampening.** Graduated to
   [harmonic_fingerprint.md](harmonic_fingerprint.md), including the
   refinement that the banks read pre-existing dimensionality rather than
   add it.
