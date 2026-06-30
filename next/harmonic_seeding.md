# Harmonic content-addressed seeding

**Goal.** Reproduce the same show from the same *music*, not the same *file*. Audio here is
lossy, so byte/file hashes break on re-encode. And we want robustness to cut-up, rearrangement,
and overlay (speech, mashup): the same musical content should still reconstruct similar
geometries even when it is perceivably different. Seed from the harmonics themselves.

**What it is (the precise name).** Locality-sensitive hashing of a perceptual harmonic
descriptor - cosine-LSH / SimHash / random-hyperplane hashing. It is a similarity search, but
NOT over geometries. It is over the *audio*; the geometry falls out deterministically from
where the audio lands. (The explicit alternative is vector-quantization: a seeded codebook of
geometry prototypes, pick the nearest by cosine. SimHash is the codebook-free streaming version
- preferred here so we ship no codebook.)

## The two layers (this distinction is the whole thing)

1. **Perceptual harmonic signature** - a small, loudness-invariant descriptor of the *recent*
   audio:
   - 12-bin **chroma** (the 64 log-bands folded onto octave-wrapped pitch classes) - timbre-
     and octave-robust, survives re-encoding, and tonal content dominates over overlaid noise.
   - a few coarse shape scalars (low/mid/high tilt, onset flux) for discrimination.
   - **L2-normalised** (kills gain/compression differences) and **EMA-smoothed** over a few
     seconds (stable, not jittery) and **LOCAL** (depends only on a recent window, so a cut-out
     segment carries its own signature).

2. **Two ways to consume it:**
   - **Discrete structure** (which scene, which behavior) via **SimHash bucket**: project the
     signature onto a fixed bank of random hyperplanes, take the signs -> a bit-bucket -> seed.
     Use only a FEW coarse bits so buckets are wide and re-encode/overlay rarely crosses a
     boundary. This SNAPS: same bucket = identical, crossed boundary = unrelated.
   - **Continuous expression** (colors, rates, the dynamics within a scene) by feeding the raw
     signature vector into the scene's modulation. This is what gives GRACEFUL similarity: a
     perturbed signature -> a slightly perturbed look, not a different one.

   The split is the answer to "similar but not bit-for-bit." Robustness lives in making buckets
   wide (the hash); the smooth part keeps the *character* (palette, motion energy) tracking the
   audio even when the discrete pick changes.

## Honestly graded

- **Re-encode (mp3<->wav, bitrate):** essentially identical - the signature barely moves.
- **Cut the song / rebuild a video:** each surviving segment reproduces its own visuals,
  because the seed is local content, not file-hash + scene-index. This is the property the byte
  fingerprint can't give. Strongest new win.
- **Speak over it / light mashup:** graceful. While the original's harmony dominates the window
  you stay in the same buckets and palette; as the overlay takes over it drifts - correctly, it
  is different content now.
- **Hard limits (be honest):** a key transposition rotates chroma (different show unless we add
  shift-invariance, e.g. CENS / the magnitude of chroma's DFT); time-stretch changes the window.
  And you can never force IDENTICAL visuals from SUBSTANTIALLY different audio without abandoning
  the premise. The guarantee is monotonic: more harmonic similarity -> more visual similarity.

Ties to the harmonic-latent / Koopman framing: the signature is a point in harmonic-feature
space, the show is a trajectory, content-addressing = neighborhoods in harmonic space map to
neighborhoods in visual space.

## The chosen shape: a live `seed_bias`, not a replacement (Ryan's refinement)

Don't pick content-addressing OR history - they both can be used. Keep every existing seed
(session identity, scene index, history/swaps) and thread a **`seed_bias`** through it: a value
that would normally be a fixed constant, but is instead driven LIVE by the harmonic channels.
The harmonics become a continuous steering signal layered onto the whole seeding fabric, so
content + history + live spectrum all coexist. The channels ARE the seed signal.

  scene_seed = session_seed ^ (index*A) ^ (swaps*B) ^ Spectrum.seed_bias()   // additive bias

Two consumption points for the SAME live signal (this is just discrete vs continuous, not a
re-dichotomy):
- **Snapshotted into discrete seeds** at the instant a thing is instanced (a scene build reads
  `seed_bias()` once, at the cut). A value used as an RNG seed avalanches, so this is where the
  harmonics "pick" structure.
- **Read continuously** where things modulate every frame (`harmonic_signature()` -> the channel
  vector). This is where graceful similarity lives - a perturbed signature gives a slightly
  perturbed look.

## Status / where it lives

- `scripts/harmonic_signature.gd` - `HarmonicSignature` (chroma + coarse shape, L2-norm, EMA;
  `vector()`, `bucket(bits)`, `seed()`). Hyperplane bank from a baked seed (global/reproducible).
- `scripts/spectrum.gd` - builds it from band centres, updates each analysed frame; exposes
  `harmonic_signature()` (continuous channel vector), `harmonic_bucket(bits)`, and
  **`seed_bias()`** (the coarse bucket spread across the seed bits, for XOR-mixing).
- `scripts/director.gd` - the auto scene seed now XORs in `Spectrum.seed_bias()` at each cut;
  prints the bucket as telemetry.

**Threaded so far ("a bunch of places"):**
- DISCRETE / seed_bias: scene build seed (`_next_entry`), transition style (`_choose_style`),
  trigger arming (`_arm`), and the novelty scene pick (`_pick_index`, draw point rotated by a
  bias fraction). Director helpers `_biased(n)` (bag index) and `_bias_frac()` ([0,1)).
- CONTINUOUS / signature: `GhostScene.chroma_hue()` -> Vector2(tonal hue, strength) from the 12
  chroma bins on the colour wheel. Used in `harmonic_lattice` (each column maps to a pitch class
  that lights it + palette pulled toward the tonal hue) and `embers` (cloud tinted toward the
  tonal hue). (`orbits` also wired but it is NOT in the SCENES catalogue, so unseen.)

Caveat to watch: live (variable fps) vs export (fixed fps) can give slightly different EMA
trajectories -> a bucket could flip at a cut; coarse buckets + smoothing keep this rare but it is
not bit-exact across the two paths.
