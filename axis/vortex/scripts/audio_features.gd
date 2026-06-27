extends RefCounted
class_name AudioFeatures

## One frame of audio, typed.
##
## This is the only thing a [VortexScene] reads from the audio engine - scenes
## never touch [AudioServer] or the analyzer directly. [Spectrum] fills one of
## these every frame and hands it to the active scene.

## Overall loudness this frame, roughly 0..1 (smoothed, soft-clipped).
var energy: float = 0.0

## Named band energies, each roughly 0..1. These are convenience views into
## [member bands] at musically useful split points.
var bass: float = 0.0
var low_mid: float = 0.0
var mid: float = 0.0
var high: float = 0.0
var treble: float = 0.0

## A short pulse near 1.0 right after an energy onset, decaying toward 0.
## Cheap stand-in for real beat detection (see README roadmap).
var beat: float = 0.0

## Spectral flux this frame: how much the spectrum *changed* (sum of positive
## per-band deltas). High when new frequency content arrives - the raw material
## for detecting a change of section.
var flux: float = 0.0

## 0..1 "we just moved into a new part of the song" score, from a sliding window
## over [member flux]. The [Director] cuts scenes on this rather than a timer, so
## changes land with the music. Stays low through a steady passage.
var movement: float = 0.0

## The full spectrum this frame: N band magnitudes, low to high, each ~0..1.
var bands: PackedFloat32Array = PackedFloat32Array()

## Seconds of audio elapsed (playback position), or an idle clock when no
## audio is loaded - so scenes still animate with nothing playing.
var time: float = 0.0


## Linear sample of the band spectrum at [param t] in 0..1, interpolated.
## Handy for spreading a scene's geometry across the whole spectrum.
func sample(t: float) -> float:
	var n := bands.size()
	if n == 0:
		return 0.0
	var x := clampf(t, 0.0, 1.0) * float(n - 1)
	var i := int(x)
	if i >= n - 1:
		return bands[n - 1]
	return lerpf(bands[i], bands[i + 1], x - float(i))
