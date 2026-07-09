shader_type canvas_item;

// Mask split - chroma-key BY COLOR, not by screen position.
//
// The source footage is lit with distinct color regions (e.g. a hard red/blue
// key-light split on a face) - the boundary moves in screen space as the subject
// moves, so this can't be a spatial crop. Every pixel is scored per LAYER: each
// layer targets one reference hue, and a pixel's membership is its hue-distance
// to that target (inside threshold = full member, feather = soft edge), gated by
// a saturation floor so near-neutral pixels (shadow, highlight, background) never
// key.
//
// LAYERS, not channels: each timeline marker contributes one layer (one color,
// one effect, one strength, one pattern placement), and the stack applies in
// chronological order - later layers act on top of earlier ones without
// disturbing them. Each layer arrives pre-weighted by its own transition
// envelope (MaskSession.layers_at) times the screen surface's presence, so every
// fade/dissolve/supersede is already encoded in u_l_w by the time pixels are
// involved. The arrays are FIXED LENGTH and always pushed in full (a partially
// assigned uniform array is silently dropped - flame.gdshader learned that the
// hard way).
//
// Effects: "erase" hides the keyed region outright. The rest are volumetric
// CONSUMING fields: the wisps themselves are the substance - where a drifting
// fbm field forms a lick of flame / vein of ice / curl of smoke, the keyed
// footage is eaten to void, each wisp edged by a deeper DARK lip (edges are
// drawn in the masking color, never the target hue - a tinted rim just paints
// the removed color back); everywhere the field is absent, the footage stays
// intact. (The sparse regions
// of the pattern DRAW THE VOID - tendrils devouring the subject - they don't
// open windows onto it.)
//
// Time comes in as a uniform (the clip's own playback position), never TIME -
// live preview and the offline export step the same clock, so a session
// reproduces its exact look frame-for-frame (same discipline as flame.gdshader).

const int MAXL = 6;   // must match MaskSession.MAX_LAYERS

// Global keying environment.
uniform float u_threshold = 0.24;  // hue-distance inside which a pixel fully belongs
uniform float u_feather = 0.12;    // softness of the membership edge
uniform float u_sat_floor = 0.18;  // below this saturation, a pixel keys to NO layer

// The layer stack (chronological; only the first u_l_count entries are live).
uniform int u_l_count = 0;
uniform float u_l_hue[6];          // each layer's target hue
uniform int u_l_effect[6];         // index into MaskSession.MASK_EFFECTS
uniform float u_l_w[6];            // envelope x intensity x surface presence
uniform vec2 u_l_off[6];           // pattern pan (unit UV)
uniform float u_l_scale[6];        // pattern zoom
uniform float u_l_dens[6];         // coverage (resonance breathe pre-folded CPU-side)
uniform float u_l_con[6];          // contrast 0..1, exponential response
uniform float u_l_glow[6];         // dark-lip depth multiplier (resonance pulse pre-folded)
uniform vec3 u_l_tdir[6];          // target hue's normalized CHROMA direction (for erase)

uniform float u_time = 0.0;        // clip playback seconds (deterministic clock)
uniform float u_aspect = 1.7778;   // video w/h, so the pattern stays isotropic

vec3 rgb2hsv(vec3 c) {
	vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Shortest distance between two hues on the wrapped 0..1 circle.
float hue_dist(float a, float b) {
	float d = abs(a - b);
	return min(d, 1.0 - d);
}

// Lattice hash on a mod-289 permutation polynomial (ported from flame.gdshader):
// inputs are bounded BEFORE any multiply, so precision holds however far the time
// scroll pushes the coordinates - a fract-of-big-product hash tiles into blocks
// within seconds of scrolling.
float permf(float x) {
	return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float hash21(vec2 p) {
	p = p - floor(p / 289.0) * 289.0;
	return fract(permf(permf(p.x) + p.y) * 0.145598);
}

float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float a = hash21(i);
	float b = hash21(i + vec2(1.0, 0.0));
	float c = hash21(i + vec2(0.0, 1.0));
	float d = hash21(i + vec2(1.0, 1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
	float v = 0.0;
	float amp = 0.5;
	for (int i = 0; i < 4; i++) {
		v += amp * vnoise(p);
		p *= 2.03;
		amp *= 0.5;
	}
	return v;
}

// One tap's membership: 1 well inside the hue target, feathered to 0 at the
// threshold edge, 0 regardless for pixels with no real color.
//
// The gate is COLORFULNESS (saturation x brightness), not HSV saturation alone.
// HSV saturation is chroma NORMALIZED BY BRIGHTNESS, so a near-black compressed
// pixel - RGB(10,25,20), say - scores "saturation 0.6" while carrying almost no
// actual color, and its quantized hue is junk that lands anywhere on the wheel
// (in red-lit footage, disproportionately on red's complement, teal). Gating on
// saturation alone let those dark chroma-blocks key at FULL strength onto any
// wide-threshold layer - the hard black block-cluster artifact, immune to
// spatial smoothing because the junk arrives in contiguous block groups. A
// pixel must have light carrying its color before its hue is trusted; dark
// regions simply cannot key, which is correct - there is no chroma information
// down there to key on.
float tap_mask(float target_hue, vec3 hsv) {
	float d = hue_dist(hsv.x, target_hue);
	float m = 1.0 - smoothstep(u_threshold - u_feather, u_threshold + u_feather, d);
	float colorfulness = hsv.y * hsv.z;
	float conf = smoothstep(u_sat_floor * 0.6, u_sat_floor * 0.6 + 0.10, colorfulness);
	// Where a pixel carries no real color, its membership is UNKNOWN - and
	// unknown leans PERMISSIVE (0.6), not zero. These masks only steer the
	// volumetrics now (erase is projection-based and gate-free), and the whole
	// philosophy of that path is that the wisp field makes the judgement calls
	// in uncertain places. Zeroing dark pixels silently confined the patterns
	// to brightly-keyed regions - on dark footage, fire configured to sweep the
	// whole scene shrank to a few faint patches.
	return mix(0.6, m, conf);
}

// This layer's membership averaged over a run of the prediction taps (see
// fragment: [0..8] = center + fine ring, [9..16] = mid ring, [17..24] = wide
// ring). Averaging the MASK across the neighborhood - never the color (averaging
// RGB across a red/blue seam invents purple) - turns the blocky steps of
// compressed chroma into coverage gradients. The mask is GREY AND CONTINUOUS end
// to end; nothing downstream may treat it as on/off.
float mask_over(float target_hue, vec3 taps[25], int start, int count) {
	float acc = 0.0;
	for (int k = 0; k < 25; k++) {
		if (k >= start && k < start + count) {
			acc += tap_mask(target_hue, taps[k]);
		}
	}
	return acc / float(count);
}

// One scalar wisp field, shaped per effect. All three share the fbm basis and
// differ in drift (how it moves), warp (how it shreds), and profile (how it's
// sharpened) - fire rises and licks, ice sits nearly still in ridged veins,
// smoke billows softly.
float wisp_field(vec2 uv, int effect, float t, vec2 off, float scl) {
	vec2 p = (uv * vec2(u_aspect, 1.0) + off) * (3.0 * scl);
	if (effect == 1) {          // fire: fast rise, hard domain warp, thinned licks
		vec2 drift = vec2(0.08 * t + 0.15 * sin(t * 0.31), -0.85 * t);
		vec2 w = vec2(fbm(p * 1.9 + drift * 0.7), fbm(p * 1.6 - drift * 0.4));
		return pow(fbm(p + drift + 1.15 * w), 1.35);
	} else if (effect == 2) {   // freeze: near-static ridged veins (crystal facets)
		vec2 drift = vec2(0.02 * t, 0.015 * t);
		float n = fbm(p * 1.4 + drift);
		float r = 1.0 - abs(2.0 * n - 1.0);
		return pow(r, 2.2);
	}
	// smoke: slow billow, gentle warp, soft gaussian coverage
	vec2 drift = vec2(0.10 * t, -0.22 * t);
	vec2 w = vec2(fbm(p * 1.3 + drift * 0.5), fbm(p * 1.1 - drift * 0.3));
	return fbm(p + drift + 0.8 * w);
}

// THE REMNANT RULE. Suppressing a keyed pixel by scaling its RGB keeps its HUE -
// a half-erased blue pixel is still blue, just dimmer, and against fully-dark
// neighbors every partial-confidence zone (mask edges, the colorfulness gate's
// ramp) renders as a halo of the exact color being removed. Removing a color
// means removing the COLOR first: chroma dies ahead of light, so the partial
// zone trends monochrome and edges land in the masking color (black), never in
// the target color.
vec3 suppressed_remnant(vec3 rgb, float s) {
	float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
	vec3 neutral = mix(rgb, vec3(lum), smoothstep(0.0, 0.5, s));
	return neutral * (1.0 - s);
}

// SUPPRESSION SHAPING - every darkening in this shader passes through here.
// Three properties, each aimed at the hard-edged transparency-mask artifact:
//   CEILING   - suppression never reaches 1.0: even a fully keyed, fully
//               consumed pixel keeps a faint ghost of the footage bleeding
//               through. Against pure black, any confidence step reads as a
//               hard line; against a ghost, the image's own continuity carries
//               across the edge.
//   CURVE     - a soft gamma lift: mid-confidence suppresses a little MORE than
//               linear, so the dark skirt extends further and more gradually
//               into uncertain zones - continuous, no knees.
//   GRANULAR  - per-pixel grain modulates the suppression, strongest where
//               confidence is ambiguous (the artifact zone), a whisper
//               elsewhere - block boundaries dissolve into grain instead of
//               lines. Clocked off the clip's own time (quantized ~9Hz), so it
//               reads as film grain and stays deterministic live vs export.
float shade_amount(float x, vec2 uv) {
	x = clamp(x, 0.0, 1.0);
	float s = pow(x, 0.85);
	float g = hash21(floor(uv * vec2(1280.0, 720.0)) + floor(u_time * 9.0) * vec2(13.0, 7.0));
	float amp = 0.06 + 0.22 * (1.0 - abs(2.0 * x - 1.0));
	s *= 1.0 - amp * g;
	return s * 0.93;
}

// Apply layer i to the running color. `w` is the layer's full on-screen weight
// (envelope x intensity x surface presence), already resolved CPU-side.
vec4 apply_layer(vec4 col, int i, vec3 taps[25], vec2 uv) {
	float w = u_l_w[i];
	if (w <= 0.001) {
		return col;
	}
	int effect = u_l_effect[i];
	if (effect == 0) {
		// ERASE IS SUBTRACTION, NOT CLASSIFICATION. Every gate-based key -
		// hue-distance x colorfulness membership - leaks at its own boundaries:
		// wherever the gate product dips (shadow edges where red drifts magenta,
		// dim reds near the colorfulness floor), pixels survive at partial
		// membership as lines and rings of the very color being erased, and no
		// amount of stacking closes them (it darkens the middle and sharpens
		// the ring). So erase doesn't ask "is this pixel red?" at all: it
		// projects the pixel's chroma onto the target color's direction and
		// SUBTRACTS that component, plus the light it carried. Linear in the
		// pixel color - no thresholds anywhere, so no boundary artifacts are
		// even possible - and pixels with no target-colored component (other
		// hues, neutrals) are untouched by construction. Chroma dies with its
		// light together: full reds land near black, faint reds go faintly
		// gray-darker, everything in between is continuous.
		float k = shade_amount(w, uv);   // strength: granular, curved, never total
		float lum = dot(col.rgb, vec3(0.299, 0.587, 0.114));
		vec3 chroma = col.rgb - vec3(lum);
		float p = max(0.0, dot(chroma, u_l_tdir[i]));
		vec3 outc = col.rgb - u_l_tdir[i] * (p * k) - vec3(0.85 * p * k);
		return vec4(max(outc, vec3(0.0)), col.a);
	}
	// Fine/regional confidence for the VOLUMETRIC masks: local coverage and the
	// whole 25-tap pyramid - a wide, spatially smooth envelope with no
	// block-scale structure (block noise is high-frequency; a ~44px window
	// doesn't see it). This is what volumetrics taper by - see below.
	float fine = pow(mask_over(u_l_hue[i], taps, 0, 9), 0.8);
	float regional = mask_over(u_l_hue[i], taps, 0, 25);
	if (max(fine, regional) * w <= 0.001) {
		return col;
	}
	float field = wisp_field(uv, effect, u_time, u_l_off[i], u_l_scale[i]);
	// Contrast: exponential response around the neutral midpoint - each half of
	// the slider is a full doubling-per-step curve, so the extremes actually
	// arrive (0 -> gamma 1/4, a washed gauze; 1 -> gamma 4, knife-edged peaks).
	float con = u_l_con[i];
	field = pow(field, pow(4.0, (con - 0.5) * 2.0));
	// Edge softness / rim strength per pattern: fire cuts thin bright licks,
	// ice glints hard along its veins, smoke stays soft and gauzy. The edge band
	// narrows with contrast, so high contrast cuts crisp, not just dark.
	float sharp = effect == 1 ? 0.10 : (effect == 2 ? 0.06 : 0.18);
	sharp *= mix(1.6, 0.4, con);
	float glow = (effect == 1 ? 0.95 : (effect == 2 ? 0.75 : 0.5)) * u_l_glow[i];
	// Tail-weighted coverage response: gentle through the working midrange,
	// accelerating hard at both ends so the slider genuinely reaches fully-sparse
	// and fully-dense (the cubic hits 0 and 1 exactly). "Fully dense" still
	// bleeds a granular ghost of the footage through - see shade_amount: total
	// suppression is where hard chroma edges come from, so it doesn't exist.
	float dens = clamp(u_l_dens[i], 0.0, 1.0);
	dens = mix(dens, 4.0 * pow(dens - 0.5, 3.0) + 0.5, 0.6);
	float thr = 1.05 - dens * 1.15;
	// The wisp IS the substance: where the field forms a lick, the footage is
	// consumed to void; everywhere else it stays intact.
	float consume = smoothstep(thr - sharp, thr + sharp, field);
	// A rim band on the intact side of the boundary - the devouring edge's lip.
	// It DARKENS (see below): edges are drawn in the masking color, never in the
	// layer's target hue - an earlier version added hue-tinted glow here, which
	// painted the removed color right back along every wisp boundary (the halo
	// complaint). It fades out entirely as coverage approaches fully-sparse:
	// "nothing consumed" must mean untouched footage.
	float rim = smoothstep(thr - sharp * 3.0, thr, field) * (1.0 - consume);
	rim *= smoothstep(0.0, 0.08, dens);
	// THE KEY DOES NOT CLIP THE PATTERN. The old model multiplied the wisp by
	// local keying confidence, so the pattern's edge WAS the key's edge - and on
	// compressed footage that edge is ragged, blocky, and temporally unstable:
	// tendrils were amputated mid-lick along compression noise, constantly.
	// Instead the volumetric draws ANYWHERE, shaped only by its own field, and
	// the key modulates its TRANSPARENCY through the wide regional envelope -
	// spatially smooth (it varies over tens of pixels, not blocks), with a soft
	// knee so it reaches full strength well inside the keyed region and tapers
	// gently across the boundary. A tendril near the edge fades on its own
	// terms; it is never cut. (The few-pixel spill past the seam is deliberate
	// and tapered - the price of never clipping, and it reads as the pattern
	// drifting across the boundary rather than an error.)
	float vol_conf = smoothstep(0.05, 0.55, regional);
	// The darkening is ONE shaped product - the wisp body and its dark lip
	// (rim folded in via max) x envelope x weight - through shade_amount
	// (granular, continuous, never total) and the remnant rule (chroma dies
	// first), so no path here can manufacture a hard edge, a pure-black pixel,
	// or a target-colored halo.
	// Freeze's rim IS its emission (below), so only fire/smoke fold the rim
	// into the dark lip - freeze's edge glows instead of charring.
	float lip = rim * clamp(0.6 * glow, 0.0, 1.0);
	float shape = effect == 2 ? consume : max(consume, lip);
	float suppress = shade_amount(shape * vol_conf * w, uv);
	vec3 outc = suppressed_remnant(col.rgb, suppress);
	// The EMISSIVE BODY - the effect's visible substance, in its own FIXED
	// palette (never the key hue, never tracing mask edges - both halo lessons
	// hold): fire glows deep-red to amber from its field's interior heat, a
	// broad body gradient placed by the field itself; freeze's crystalline
	// VEINS shine cold blue-white - the rim band is the vein, the effect's
	// defining line-work, bright and distinct even at sparse coverage; smoke
	// stays pure consumption. Resonance pulses this through the glow factor.
	// An effect that only darkens is invisible on dark footage - fire and ice
	// must carry their own light.
	float heat = smoothstep(thr, thr + 0.4, field);
	vec3 emiss = vec3(0.0);
	if (effect == 1) {
		emiss = mix(vec3(0.55, 0.08, 0.01), vec3(1.0, 0.55, 0.10), heat) * heat * 0.9;
	} else if (effect == 2) {
		emiss = vec3(0.62, 0.82, 1.0) * (rim * 1.1 + pow(heat, 2.0) * 0.3);
	}
	outc += emiss * (glow * vol_conf * w);
	return vec4(outc, col.a);
}

void fragment() {
	vec4 src = texture(TEXTURE, UV);
	// Prediction taps: compressed video stores chroma at quarter resolution in
	// quantized blocks, so at hard color edges the hue channel steps in whole
	// blocks - keyed from the center pixel alone, entire blocks flip fully
	// in/out and the effects render as hard black squares along seams. A small
	// ring of taps (converted to HSV once, shared by every layer) lets each
	// layer's mask read as neighborhood COVERAGE instead - see chan_mask.
	vec3 taps[25];
	taps[0] = rgb2hsv(src.rgb);
	vec2 rad = TEXTURE_PIXEL_SIZE * 2.5;
	vec2 rad_mid = TEXTURE_PIXEL_SIZE * 7.0;
	vec2 rad_wide = TEXTURE_PIXEL_SIZE * 22.0;
	for (int k = 0; k < 8; k++) {
		float a = float(k) * 0.7853981634;   // TAU / 8
		vec2 dirv = vec2(cos(a), sin(a));
		taps[k + 1] = rgb2hsv(texture(TEXTURE, UV + dirv * rad).rgb);
		// Outer rings staggered so the spokes interleave instead of aligning.
		float ab = a + 0.3926990817;         // TAU / 16
		taps[k + 9] = rgb2hsv(texture(TEXTURE, UV + vec2(cos(ab), sin(ab)) * rad_mid).rgb);
		float aw = a + 0.1963495409;         // TAU / 32
		taps[k + 17] = rgb2hsv(texture(TEXTURE, UV + vec2(cos(aw), sin(aw)) * rad_wide).rgb);
	}
	vec4 col = src;
	for (int i = 0; i < MAXL; i++) {
		if (i >= u_l_count) {
			break;
		}
		col = apply_layer(col, i, taps, UV);
	}
	COLOR = col;
}
