# Where the Value Hides: universal approximation, the three equivalences, and saying it before the ablations close

Status: part-grounded, part-conjecture (2026-06-04). The structural argument is
real and now carries a paper home (the Discussion fragment
`architecture-separation` + Proposition `proof-architecture-separation`, with
Telgarsky and Eldan-Shamir as the spine). What stays open - and is honest about
being open - is the empirical step: that comparable-capacity architectures show
*separated error boundaries under training*, which needs ablations we have not
run. Sibling to [forced_computation.md](forced_computation.md),
[the_fifth_dimension.md](the_fifth_dimension.md),
[observer_frequency.md](observer_frequency.md).

## The objection, stated at full strength

Math is largely the business of proving equivalences. Universal approximation
already provides a strong foundational one: a wide enough network of essentially
any family can approximate any reasonable function. So if even an elaborate Praxis
experiment is, by hypothesis, emulable by some other network - decomposable,
re-expressible, reachable by a different mesh - then *where is the value in the
architectural complexity*? If everything is equivalent in the limit, the geometry
is decoration.

This is the strongest form of the skeptic's case, and the right move is not to
dodge it but to notice that it smuggles in a false step. It treats "equal as sets
of representable functions" as if it meant "equal as engines you can train." It
does not.

## The three equivalences (the load-bearing distinction)

Universal approximation lives at exactly one layer and is silent on the two that
matter:

1. **Representational equivalence.** *There exist* parameters making architecture
   A approximate function f. This is all UAT gives - an existence statement,
   quantified over parameters at unbounded width.
2. **Learnability equivalence.** SGD, from a random initialization, actually
   *reaches* that f in feasible compute and data. UAT says nothing.
3. **Statistical equivalence.** Two architectures *generalize* equally at a fixed
   sample budget. UAT says nothing here either.

The entire value of an architecture lives in layers 2 and 3. Proving layer 1 about
two families tells you nothing about whether one of them finds a good solution
before the heat death of the universe, or whether the solution it finds holds up
on data it has not seen. "A solution exists" and "you can find a good one cheaply"
are different theorems; UAT is only the first.

## The math is on our side, not theirs

The part worth internalizing: the foundational mathematics does not say
architecture is neutral. It says the opposite, *provably*.

- **Depth separation.** Telgarsky (2016) constructs functions that a deep, narrow
  network represents exactly but that any shallower network needs *exponential*
  width to approximate. Eldan-Shamir (2016) give a function a depth-three network
  expresses with polynomial width that any depth-two network needs exponential
  width to match. Same representable-function set in the limit; representation
  *cost* separated by 2^Ω(k). So "architecture is just a constant factor" is not a
  cautious null - it is a claim already falsified.
- **Implicit bias / lottery tickets.** Two architectures (or two
  initializations) of identical capacity induce different *distributions over
  which solution SGD lands in*. Approximation theory is silent on this entirely;
  it is about which functions exist, not which one you converge to. The real
  object of study is the triple `(architecture, optimizer, data) -> a prior over
  learned functions`, and that prior is the inductive bias the geometric heads are
  built to shape.

The weakest link in the skeptic's steelman is "decomposes to a linear function /
emulated by some other function." Composed nonlinearities are not linear - if they
were, you would not have needed UAT in the first place, you would use a matrix. The
honest version is "emulated by some *other network*," and the instant that is said,
the only remaining question is *at what cost*, which is precisely the
separation question the theorems answer.

## Saying it before we've proven it

The user's framing: "ablations are expensive, so we're going to have to say it
before we've proven it." That is legitimate, and it is the same discipline the
paper already runs everywhere else - state the conjecture, in the conjecture's
register, with whatever spine the existing math gives it.

What is *established* (cite, don't re-derive): representation cost is
architecture-dependent and can be exponentially separated. What is *conjectured*
(the framework's measurable edge): that comparable-capacity architectures, under
training, exhibit **separated generalization-error boundaries across
configurations** - that the inductive-bias gap survives the descent from
representation cost to learned function. The falsifiable form:

> Across architectures of comparable parameter count and compute, there exist
> task/data regimes where generalization error is separated, and the separation
> is a property of inductive bias, not capacity.

The ablation that would close it: fix budget (params, FLOPs, tokens), vary the
geometry (harmonic vs flat head, coupled vs sequential, crystal vs free
classifier), sweep task families by combinatorial structure, and measure whether
the error boundaries are separated and whether the separation tracks the structure
rather than the count. Until that is run, the paper says exactly this much and no
more: the cost separation is mathematics; the learned-error separation is the
prediction. The seam is marked, not hidden - which is the only thing that earns
the right to state a conjecture out loud.

## The deeper reframe (voice)

There is a reading underneath this that the paper should *not* assert but is worth
keeping: if value lives in learnability and generalization rather than
representability, then "what a model can know" is not a fact about the function
class at all - it is a fact about the *path* through it that a given geometry makes
cheap. Two architectures with the same reachable set still live in different
worlds, because the worlds are made of *what is cheap to reach*, not what is
possible to represent. That is the same observer-relative move the paper makes for
tractability, here applied to capability itself: the space of functions is shared;
the metric on it - which ones are near, which are exponentially far - is what the
architecture chooses. Universal approximation flattens that metric to a single
bit (in / out of the closure) and then declares all architectures equal under it.
The whole argument of Praxis is that the metric, not the bit, is where modeling
happens.

## Why this is a note and not (all) paper content

The grounded spine - three equivalences, the depth-separation witnesses, the
honest open ablation - is now in the Discussion as `architecture-separation`,
because it is defensible and load-bearing. The metric-not-the-bit reframe and the
"different worlds made of what is cheap to reach" telling stay here, in voice,
behind the same fence the observer thread keeps. The paper gets the part that can
be attacked on its merits; the note holds the part the work gestures toward.
