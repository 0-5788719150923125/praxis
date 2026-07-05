extends RefCounted
class_name HarmonicSignature

## HarmonicSignature - a perceptual, loudness-invariant descriptor of the RECENT audio, plus a
## cosine-LSH (SimHash) seed derived from it. The point: seed the show from the *harmonic
## content present right now*, not from the file - so the same music reconstructs the same
## visuals even when the audio is re-encoded (lossy), cut apart, or has other sound laid over
## it. Similar harmonics -> similar descriptor -> same hash bucket -> same seed, with graceful
## drift as the content diverges. Full design in next/harmonic_seeding.md.
##
## Descriptor (DIM floats) = a 12-bin CHROMA (the 64 log-bands folded onto octave-wrapped pitch
## classes, so it is timbre- and octave-robust and survives re-encoding) + a few coarse shape
## scalars (low / mid / high tilt, onset flux) for discrimination. It is L2-NORMALISED (gain
## and compression don't matter) and EMA-SMOOTHED over a few seconds (stable, not jittery) and
## LOCAL (depends only on a recent window, so a cut-out segment carries its own signature).
##
## Two ways to consume it, and the distinction is the whole point:
##   vector()      - the continuous descriptor, for SMOOTH modulation of a scene's dynamics
##                   (a perturbed signature -> a slightly perturbed look). Graceful similarity.
##   bucket(bits)  - a SimHash: the descriptor's sign against a FIXED bank of random hyperplanes
##                   (baked seed, so the hash is identical on every machine). For DISCRETE
##                   choices (which scene / behavior). Snaps: same bucket = identical, a crossed
##                   boundary = unrelated - so keep `bits` small to make the buckets wide.

const DIM := 16
const BITS := 32
const PLANE_SEED := 4099   # fixed: the hyperplane bank is GLOBAL / reproducible, not per-session
const REF_FREQ := 16.35    # C0, for pitch-class folding
const TAU_SMOOTH := 2.5     # default EMA time constant (seconds of context integrated)

var tau := TAU_SMOOTH                # this instance's EMA time constant (see _init)
var _vec := PackedFloat32Array()     # current smoothed, normalised descriptor (DIM)
var _pc := PackedInt32Array()        # band index -> pitch class (0..11), precomputed
var _planes: Array = []              # BITS hyperplanes of DIM floats (fixed)


## [param smooth_tau] trades stability for reaction time: the default suits seeding
## and slow modulation; a listener that must NOTICE a change quickly (the [Echo]
## re-localizer) runs a second, faster instance.
func _init(band_centres: PackedFloat32Array, smooth_tau := TAU_SMOOTH) -> void:
	tau = maxf(0.05, smooth_tau)
	_vec.resize(DIM)
	_pc.resize(band_centres.size())
	for i in band_centres.size():
		var f: float = maxf(band_centres[i], 1.0)
		_pc[i] = int(posmod(round(12.0 * log(f / REF_FREQ) / log(2.0)), 12.0))
	# Fixed gaussian hyperplanes from a baked seed -> the SimHash is the same everywhere.
	var r := RandomNumberGenerator.new()
	r.seed = PLANE_SEED
	for k in BITS:
		var pl := PackedFloat32Array()
		pl.resize(DIM)
		for d in DIM:
			pl[d] = r.randfn(0.0, 1.0)
		_planes.append(pl)


## Fold this frame's bands into the descriptor (chroma + coarse shape), normalise, and ease the
## smoothed vector toward it by `dt`. Call once per analysed frame.
func update(bands: PackedFloat32Array, low: float, mid: float, high: float, flux: float, dt: float) -> void:
	var raw := PackedFloat32Array()
	raw.resize(DIM)
	for i in mini(bands.size(), _pc.size()):
		raw[_pc[i]] += bands[i]                  # octave-fold into 12 pitch classes
	raw[12] = low
	raw[13] = mid
	raw[14] = high
	raw[15] = flux
	var n := 0.0                                 # L2 normalise -> loudness invariance
	for d in DIM:
		n += raw[d] * raw[d]
	n = sqrt(maxf(n, 1e-9))
	var k := 1.0 - exp(-dt / tau)
	for d in DIM:
		_vec[d] = lerpf(_vec[d], raw[d] / n, k)


## The continuous descriptor (DIM floats), for smooth dynamics modulation.
func vector() -> PackedFloat32Array:
	return _vec


## SimHash bucket from the first `bits` hyperplanes (wider buckets = more robust). Identical
## descriptors give identical buckets; perceptually close ones usually share a bucket.
func bucket(bits: int) -> int:
	var s := 0
	for k in mini(bits, BITS):
		var dot := 0.0
		var pl: PackedFloat32Array = _planes[k]
		for d in DIM:
			dot += _vec[d] * pl[d]
		if dot >= 0.0:
			s |= (1 << k)
	return s


## The full SimHash, as a content-derived seed.
func seed() -> int:
	return bucket(BITS)
