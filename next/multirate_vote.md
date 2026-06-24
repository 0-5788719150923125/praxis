# Multirate Vote: Per-Band Nyquist Sampling, Serpent as Rate Controller, and the FiLM→Cross-Attention Bridge

> Status: **active design thread (2026-06-23).** Born from a chain: "would 44K
> vote samples help CALM?" → no (Monte Carlo on a biased estimator) → *but* the
> signal-processing reading of the question is real. Ties together the harmonic
> head, ArcHoPE/Serpent ([[project_learnable_rope_theta]] + `praxis/encoding/archope.py`),
> the prismatic fast-weight overlay ([[project_prismatic_fast_weights]]), the CALM
> patch-vote ([[project_calm_sampler_brier_fix]]), and the amp-modulation work
> ([[project_harmonic_amp_modulation]]). Sibling to the belief-state regularizer item.

## The one line

**If the harmonic latent is a bandlimited signal, then sample-rate is a per-band
Nyquist quantity, not a global draw count - so allocate a fixed vote budget across
spectral bands by their bandwidth (Serpent sets each band's rate, the amp envelope
sets its budget), and reach cross-attention cheaply by *querying the fast-weight
state you already maintain* rather than bolting on softmax attention.**

## Part 1 - the multirate vote (the rigorous form of "44K samples")

The naive scale-up (uniform 500 → 44,000 draws) fails twice: Monte Carlo error
falls as `1/sqrt(N)` so the categorical vote-winner has already converged at 500,
and - worse - the energy head currently samples the *marginal* not the
*conditional* ([[project_calm_bottleneck_diagnosis]]), so more draws sharpen a
biased estimator (harder collapse onto "the the the"). More draws is variance
reduction on the wrong distribution.

The signal-processing reading rescues the intuition. The harmonic head **is** a
spectrum (amplitudes over frequency bands), and a signal's required sample rate is
set by its highest active frequency. So the right object is **multirate / subband
coding**, not a global rate:

- **Near-DC bands** (the bias / stable-geometry content) want a rate ≈ 1 -
  sampling them densely is pure waste.
- **High-frequency bands** (the variance / fast-changing content) are the only ones
  that want audio-like density.

So: keep the same ~500-draw budget but **stratify it across bands by bandwidth**
instead of spreading it uniformly - importance sampling of the vote. This is the
defensible "44K where it matters, 1 where it doesn't," at flat cost. Cousin of the
WaveletLM subband block ([[project_wavelet_block]]).

## Part 2 - Serpent as the per-band rate controller

ArcHoPE's per-band warp `u_i = p + rho_i·t` already assigns each band a different
*effective frequency* over the (position, depth) diagonal - it is **already a
per-band rate dial**, just read today for phase, not for sampling. Reuse it: let
`rho_i` set band `i`'s Nyquist rate, and let the amp-modulation envelope `f_t`
([[project_harmonic_amp_modulation]]) set its draw *budget* (allocate draws ∝
amplitude×bandwidth). Open direction to decide empirically: do gradient-starved
low-amplitude bands get *fewer* draws (don't waste budget on dead bands) or *more*
(invest to wake them)? - the amp envelope makes either policy a one-line weight.

## Part 3 - the cheap path to cross-attention (FiLM → linear attention)

The energy/flow head conditions via **adaLN on `h_cond`** = FiLM: global
modulation, no per-position selection. True cross-attention is per-query selection
over context. The cheap bridge is already in the tree: the **prismatic3 fast-weight
overlay** (`u_t ⊗ v_t` delta-rule, [[project_prismatic_fast_weights]]) *is* linear
attention - a rank-r outer product is precisely a linear-attention KV state. So the
path from FiLM → cross-attention is **"let the decode *query* the fast-weight state
you already maintain,"** not "add softmax attention + a KV cache." Linear/Infini
attention is the cheap on-ramp; softmax cross-attention is the expensive version we
don't need to start with. Same primitive cross-attends belief states across
parallel models at the blending layer (the belief-state regularizer item).

## Part 4 - what cross-attention is *for*: multi-view modality inputs

Cross-attention needs multiple *views* of the same content to attend across. The
cheap-path bet is that every view is a **fixed data transform** (FixedCodec
philosophy, [[project_calm_fixed_codec]]) - no learned vocoder, no vision backbone,
so it stays inside the dependency rule and adds no audio/vision stack:

- **Audio view.** Render text → speech via plain TTS, then pass the waveform
  through a *standard set of audio modulators* (pitch/formant/time-stretch/filter -
  ordinary DSP transforms on voice). The audio articulates the same text the token
  stream carries; the model cross-attends the two. The harmonic head is *already a
  spectrum*, so audio is the most natural second modality - same representation.
- **Vision view.** Print text onto a pixel grid (a standard JPG with text on it),
  with **intelligent zoom + masking** to keep the input tiny. Then *sample* it with
  **zoom as the amplifier**: a foveated/importance sampler that spends its ~500
  samples where detail lives reconstructs the whole image closely from few draws.
  This is progressive-JPEG / wavelet-coefficient-thresholding made adaptive - the
  established part is the reconstruction; the open part is *learning where to look*
  (the importance map). Directly realizes the dormant ArcHoPE "recognize text
  rendered as pixels, blend via cross-attention" roadmap fragment.
- **Video view (the horizon).** Frames over time, where **harmonics = structure**
  manifests as a *projected geometry* reconstructed from samples - reality
  converging out of scattered draws like a **progressive Blender render** (pixels
  splattered everywhere, then resolving). This is the same mechanism as the energy
  head's sample-and-converge and the "spider raised one slice of pi at a time"
  continuous-extraction framing - sampling *is* reconstruction, and more samples
  (or better-allocated ones, Part 1) sharpen the render. Time is the slow clock,
  the harmonic spectrum the structure carried across frames.

The unifying claim: **sampling-as-reconstruction is one mechanism wearing three
modality costumes.** The patch-vote that builds a latent, the foveated sampler that
rebuilds an image, and the frame sampler that resolves a video are the same
draw-and-converge loop; multirate (Part 1) is how you spend the budget; Serpent
(Part 2) is the per-band rate; the fast-weight query (Part 3) is how the modalities
attend to each other.

## Honest blockers (the gates, not footnotes)

- **Near-DC spectrum gate.** Multirate buys nothing until bands carry *distinct*
  content. The field is near-DC today (the flat-spectrum gradient-starvation
  problem amp-modulation is fighting). Verify the head-snapshot spectrum is
  decaying-but-non-trivial *before* building the multirate vote - no high band, no
  oversampling payoff.
- **"Reconstruct from 500 samples" rests on the importance map.** Progressive
  reconstruction is established (wavelet thresholding, progressive JPEG); the
  *learned where-to-look* (zoom-as-amplifier foveation) is the unproven, load-
  bearing part. Cheapest probe: a *fixed* foveation (log-polar / center-weighted)
  first, learn the map only if the fixed one underperforms.
- **Modality alignment + tiny-scale doubt.** Cross-attending text↔audio↔pixels
  needs the views time-aligned, and small undertrained models may not have the
  capacity to exploit multi-view signal at all - test the cheapest single extra
  modality (audio, same spectral representation) before vision/video.
- **Scope discipline.** Audio modulators and pixel-rendering are fixed transforms
  (good); the moment any of this wants a learned vocoder or a vision backbone it
  has left the cheap path and must be re-justified against the dependency rule.

## Cheapest first probe

Don't build the multimodal stack. First: (1) confirm the spectrum is non-degenerate
on a current checkpoint (head snapshots); (2) implement the stratified per-band vote
allocation as a generation-config option over the *existing* patch-vote (same total
budget, Serpent-weighted) and A/B coherence; (3) only then prototype the single
cheapest extra view (audio-spectrum cross-attention via the fast-weight query),
leaving vision/video as the documented horizon.
