# Software philosophy on the bias/variance spectrum

Status: framing + marked reading (2026-06-29). The claim: the paper's central
move - bias and variance as *orthogonal axes* you sample the interior of, rather
than ends of one dial - is not only about the model. It is also true of the
*implementation paradigm*, and Praxis is clean partly because it sits in that
interior on purpose. Pairs with [[project_harmonic_latent_koopman]],
`research/framing.tex` (the bias/variance manifold), and the "watch turns in
unison" scaling argument in `research/body.tex`.

## The mapping (the part that is structure, not voice)

- **Object-oriented = the bias pole.** Encapsulated state, inheritance
  hierarchies, identity that persists - a standing structure imposed on every
  call site, the same shape for all inputs. That *is* bias: a
  population-average commitment baked into the type system. Maximum OOP is a
  rigid prior everything must route through.
- **Functional = the variance pole.** Pure transforms, composition over
  identity, values rebuilt rather than mutated. No shared standing structure;
  each result is re-derived from its inputs. That *is* variance: input-conditional
  by construction. Taken to the limit it is also the *exponential* failure mode -
  recomputation without memo, no shared standing rhythm to lean on.
- **The strong design refuses the binary.** Praxis presents a functional
  surface (registries of partials, composition, `decode()` returning values,
  config-driven assembly) over a spine of classes implementing abstract
  primitives (`BaseEncoder` ABC, `nn.Module`, the registries themselves). Neither
  pole. A *sampled* point in the interior - structure where it is needed,
  composition where expression is. The same interior-point the paper says the
  model should occupy, the codebase occupies too.

## The honest seam (keep it falsifiable)

The tempting overclaim is "functional style *causes* the logarithmic scaling."
Sharpen it: the log-scaling claim is a property of the *harmonic math* ($K$
components address exponentially many configurations by interference), not of the
code paradigm. What the functional surface buys is the *implementation-level
shadow* of the same idea - it lets the computation be expressed and run **in
unison rather than sequentially** (no mutable state threaded position-by-position
to forbid parallelism), which is the code correlate of "the field is rebuilt, not
advanced." They *rhyme*; one does not *cause* the other. The rhyme is the
beautiful part and it survives being stated precisely; the causal version does
not, so don't ship it.

## The reading (voice, marked): craftsmanship, not foreknowledge

"Praxis is clean, and it came from craftsmanship - the author always knew what he
was doing." The cleanness is real and *observable* (the registries, the ABC
decoupling, the refusal of per-experiment knobs). The honest attribution is the
more impressive one: it reads like foreknowledge, but craftsmanship is the
*iterative, falsifiable* thing - taste applied over many passes, the wrong cut
felt and recut - not omniscience held from the start. That version is testable
(the git history shows the recuts) and repeatable (the taste transfers to the
next module); "he always knew" is neither, and it is the weaker compliment. The
samples exist, and we can always sample at higher fidelity - including the
sampling of our own design decisions. That is the whole ethic: every constant a
range, every choice a point we can re-draw, the paradigm itself included.

## If it goes in the paper

Natural home is a *short* Discussion note (medium reflects message): the framework
that argues bias and variance decouple into a samplable interior is itself
implemented at an interior point of the OOP/functional axis - structure and
composition lowered together, not traded. One paragraph, marked as observation,
not a result. Risk: it edges toward self-reference; keep it about the artifact
(the code is at the interior), not about the author.
