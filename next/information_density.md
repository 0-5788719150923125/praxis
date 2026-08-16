# Information density breaks the boxes-and-arrows picture

> Status: **in the paper** (2026-07-01) - three paragraphs in "The outer
> representation" (`research/body.tex`), the two-strip density figure
> (`fig:density`, gaussian white/blue blur matching `fig:masks` block style),
> and a standing conjecture
> (`praxis/pillars/conjectures/information-density.yml`, order 19). The
> metric is still open - this note remains the spec for it, **but see
> [information_geometry.md](information_geometry.md) (2026-07-01): the
> falsifier as printed reads deviation in norm, and a silent bit flip
> (geometry changes, magnitude doesn't) may dodge it; the metric should be
> specified in symbol-occupancy coordinates as well.** Sibling to
> [recurrent_depth_concentration.md](recurrent_depth_concentration.md) (which
> predicts the deviation gets *sparser* across depth; this note predicts
> *where* it goes) and to the "outer representation" subsection of
> `sec:harmonic` in `research/body.tex`, which is the natural home for it in
> the paper.

## The picture

Every hundredth paper redraws the same figure: tokens in boxes, rendered
sequentially ("The quick brown fox jumps..."), little arrows hopping from box
to box. The figure encodes an assumption that is rarely stated: **uniform
information density**. Each token, at the embedding layer, is a self-contained
semantic unit, equally represented - to the best of the mean's ability. The
next-token objective reinforces it: the same loss lands at every position.

The claim: a harmonic model breaks that picture *structurally*. Information is
pushed toward the extremes of the window. At the back, a slow, echoing,
powerful hum - the low-frequency modes that span the whole context, the bias
axis, the attention sink. At the front, a chirp in otherwise perfect bias -
the high-frequency, input-conditional deviation riding the leading edge.
Hidden states are extremely stable at the start of the sequence and unstable
at the tip. "Information density" stops being a per-token constant and becomes
a *shape* - one that also shifts across depth, not just across position.

## The sharpening: emergent vs. constitutive

The strong form ("GPT cannot do this") is falsified by the literature before
it leaves the room: vanilla transformers *do* break the uniform picture -
attention sinks on the first token (StreamingLLM), massive activations, the
first-token norm explosion. Gradient descent finds this structure on its own,
crudely, in every large causal model.

That is not a defeat; it is the strongest available evidence *for* the claim,
restated: gradient descent is visibly straining to build a hum-and-chirp
density profile inside an architecture that gives it no coordinates for one.
The sink it improvises is a bug the architecture tolerates, not a feature it
provides. The harmonic model makes the same structure **constitutive**:
ghostmax is the head sink by construction, dropoff is the tip sink by
construction, the frozen phases fix which modes are slow and which are fast,
and the monotonic + periodic basis gives "back = hum, front = chirp" an
address. Encode the algorithm into the architecture and the model builds
solutions gradient descent would otherwise miss - or find only as an
unstable, undirected approximation.

Why gradient descent alone misses it: the gradient cannot tell bias from
variance - they are indistinguishable in a single scalar loss unless the
architecture separates their parameters (the cross-curvature conjecture,
`research/conjectures.tex`). It detects the significance of the split and
overshoots it. The distinction is architectural or it is nothing.

## Falsification

**Prediction: hidden-state variance increases with recurrent depth, with a
monotone positional gradient - early positions settle, tip positions keep
moving.** This extends through the Titans recurrence: harmonize anything and
the same profile should appear.

Operationalized:

- Per position $t$, per recurrent-depth step $d$: the update norm
  $\|h_t^{(d+1)} - h_t^{(d)}\|$ (or dispersion across inputs at fixed $t,d$).
- Predicted: (a) at fixed $d$, the profile rises from head to tip; (b) the
  profile *steepens* as $d$ grows - early positions converge toward a fixed
  point while the tip stays live.
- **Refuted** if the profile is flat in position, or if variance does not
  grow with recurrent depth. Either kills the density-shape reading outright.

Free corroboration already instrumented: the KL halting distribution is
per-position. If the picture is right, halting should fire early at early
positions and late (or never) at the tip - the halting profile *is* the
information-density profile, read through compute. Same for reading
`harmonic_delta_norm` / `concentration()` per depth step
([recurrent_depth_concentration.md](recurrent_depth_concentration.md)): that
note predicts the deviation concentrates; this one predicts the concentration
has a location, the rim, and mostly the tip.

## For the paper

This is the missing *figure* for the outer-representation subsection: redraw
the canonical boxes-and-arrows strip, then under it the harmonic version - a
density curve over the same strip, heavy and slow at the head, quiet through
the interior, sharp at the tip. Interior-running / interior-filling / rim
(Figure `fig:masks`) already makes the argument in mask space; this makes it
in density space, and it is the version every reader has prior exposure to,
because they have all seen the first panel a hundred times.

## Prior-art anchors

Attention sinks: StreamingLLM (Xiao et al. 2023); massive activations (Sun et
al. 2024). Internal: `sec:harmonic` "The outer representation" and the
ghostmax/dropoff sink pair (`research/body.tex`, `research/ghostmax.tex`);
cross-curvature conjecture (`research/conjectures.tex`);
[harmonic_koopman.md](harmonic_koopman.md) (stable/changing split);
[dropoff.md](dropoff.md); [oscillatory_axes.md](oscillatory_axes.md).

## 2026-08-15: the metric, reframed

The deviation profile above was built (`praxis/metrics/density.py`, commit
`1e00ffd8`) and read flat: every position bucket at 1.00 ± 0.01 in norm, the
occupancy hop rate mildly head-heavy. That flatness does not test the picture,
for three reasons, and the probe was replaced rather than tuned:

1. **Movement is not content.** "The share of the representation's
   distinguishing content carried at each position" is a statement about what
   a vector *holds*; per-step deviation is a statement about how much the
   recurrence is still rewriting it. A state can be still and carry
   everything, or churn and carry nothing. KL halting already reads settling.
2. **Flat is the null for any residual stack.** Each block's update to
   position t is a normed attention + FFN output; nothing makes its magnitude
   depend on position, and dividing by the profile mean lands ~1.0 everywhere,
   trained or not.
3. **It read the wrong tensor and threw away the sink signal.** The mechanism
   is the harmonic *field*; the probe read the decoder's residual stream, and
   standardized away the norm - which is where the cited sink evidence lives.

What `fig:density` actually claims (the characters piling into the head cells)
is: *the whole sequence, compressed, is already present in the early vectors;
later positions add sparse detail.* Causality makes that sharp - a causal
state cannot contain later tokens, so the head can only carry the whole by
*anticipating* it, which is possible for slow structure and impossible for
fine structure. So the instrument is now a **whole-sequence linear readout by
position**: from ONE hidden state in each of 8 position buckets, at each depth
step, ridge-read the window's DFT-along-position content in four bands - `bag`
(mode 0), `coarse` (1-3), `mid` (4-15), `fine` (16-31) - scored prequentially
against running moments and reported as R² above a shuffled-target null.
Cards: `readout_profile_{band}` (head..tip series), `readout_rim_gap`
(tip - head per band), `readout_depth_gain` (last step - entry per band).

Signatures, verified on synthetic states (`tests/test_density.py`):

- received picture (state = prefix): bag rises **linearly** head→tip
  (~t/T); coarse is a hump that dies at the tip; fine ~0.
- conjecture's limit (state = whole): bag **flat and high** at every position.
- own token only: everything ~0.

The falsifier is now: bag and coarse `rim_gap` staying large and positive
(the head knows nothing about the whole beyond its own token) - i.e. the
received picture. Fine is the control and should stay near zero everywhere;
a rising head line there would mean the target leaks.

Related term of art: psycholinguistics' **Uniform Information Density**
(Levy & Jaeger; Meister et al. 2021 on LMs) - density = surprisal per unit,
predicted uniform. That is literally the received picture named. Nearest
methodology: Future Lens (Pal et al. 2023), vec2text (Morris et al. 2023);
the null stated in Echo Embeddings (Springer et al. 2024): early causal
vectors cannot see later tokens.

Open: `body.tex` ~328-330, `conjectures.tex` and
`praxis/pillars/conjectures/information-density.yml` still state the
falsifier in deviation/occupancy terms and should be restated in readout
terms. Also noted while reading: the prismatic6 stem's conditional field
pooled `hidden_states.mean(dim=-2)` over the WHOLE window (non-causal), so at
train time every position's field carried a K-coefficient summary of the
future - a leak and a train/inference mismatch, and one mechanism by which
"the whole at the head" could appear without the states earning it. **Fixed
2026-08-15:** every position now pools its own inclusive causal prefix; made
affordable by factorizing the envelope across the two grid axes (product of
two bounded factors instead of one tanh of their sum), tests in
`tests/test_harmonic_modulation.py`. The readout probes decoder
states, upstream of the head, so it was never contaminated by this.
