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
// footage is eaten to void, rimmed with a glow in the layer's own target hue;
// everywhere the field is absent, the footage stays intact. (The sparse regions
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
uniform float u_l_glow[6];         // rim glow multiplier (resonance pulse pre-folded)

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

vec3 hsv2rgb(vec3 c) {
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
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
// threshold edge, 0 regardless for near-neutral pixels.
float tap_mask(float target_hue, vec3 hsv) {
	float d = hue_dist(hsv.x, target_hue);
	float m = 1.0 - smoothstep(u_threshold - u_feather, u_threshold + u_feather, d);
	return m * smoothstep(u_sat_floor, u_sat_floor + 0.08, hsv.y);
}

// This layer's membership averaged over a run of the prediction taps (see
// fragment: [0..8] = center + fine ring, [9..16] = broad ring). Averaging the
// MASK across the neighborhood - never the color (averaging RGB across a
// red/blue seam invents purple) - turns the blocky steps of compressed chroma
// into coverage gradients. The mask is GREY AND CONTINUOUS end to end; nothing
// downstream may treat it as on/off.
float mask_over(float target_hue, vec3 taps[17], int start, int count) {
	float acc = 0.0;
	for (int k = 0; k < 17; k++) {
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
vec4 apply_layer(vec4 col, int i, vec3 taps[17], vec2 uv) {
	float w = u_l_w[i];
	// Fine confidence: local coverage, mildly lifted so uncertain edges lean
	// toward the effect. Broad confidence: the wider ring's regional read -
	// "this AREA is basically keyed, whatever this block says" - the prediction
	// the volumetrics use to render over artifacting (see below).
	float fine = pow(mask_over(u_l_hue[i], taps, 0, 9), 0.8);
	float broad = (fine * 9.0 + mask_over(u_l_hue[i], taps, 9, 8) * 8.0) / 17.0;
	if (max(fine, broad) * w <= 0.001) {
		return col;
	}
	int effect = u_l_effect[i];
	if (effect == 0) {
		// erase: darken this layer's pixels by weight, from the FINE confidence
		// alone - erase has no field of its own to justify reaching over
		// uncertainty, so its boundary stays honest. The darkening runs through
		// shade_amount: granular, continuous, never total. RGB only, alpha
		// untouched - this surface is drawn OVER the raw video, so a fading
		// alpha would let the raw color bleed back through the "erased" region.
		return vec4(col.rgb * (1.0 - shade_amount(w * fine, uv)), col.a);
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
	// A rim band on the intact side of the boundary - the devouring edge's
	// luminous lip, licking into the footage. It fades out entirely as coverage
	// approaches fully-sparse: "nothing consumed" must mean untouched footage,
	// not footage wearing free-floating glow.
	float rim = smoothstep(thr - sharp * 3.0, thr, field) * (1.0 - consume);
	rim *= smoothstep(0.0, 0.08, dens);
	vec3 tint = hsv2rgb(vec3(u_l_hue[i], 0.75, 1.0));
	if (effect == 2) {
		tint = mix(tint, vec3(1.0), 0.45);   // ice glints toward white
	}
	// THE JUDGEMENT CALL. Keying confidence is grey and continuous - but printed
	// linearly into a wisp, every blocky fluctuation of compressed chroma
	// renders as stepped darkness, and a tendril crossing an uncertain zone
	// fragments along block boundaries. So where the FIELD is present, the
	// layer commits: confidence is allowed to rise to the broad regional read
	// (the wisp completes itself over local artifacts), and mid-grey confidence
	// inside a strong wisp is squeezed toward covered (smoothstep) - artifact-
	// prone places get covered by the volumetric rather than exposed by it.
	// Where the field is absent, confidence stays untouched - the judgement is
	// the wisp's to make, never a blanket dilation of the key.
	float conf = max(fine, broad * consume);
	conf = mix(conf, smoothstep(0.10, 0.60, conf), 0.85 * consume);
	// The final darkening is ONE product - field x confidence x weight - shaped
	// by shade_amount (granular, continuous, never total), so no path through
	// this branch can manufacture a hard edge or a pure-black pixel.
	float suppress = shade_amount(consume * conf * w, uv);
	vec3 outc = col.rgb * (1.0 - suppress) + tint * rim * glow * conf * w;
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
	vec3 taps[17];
	taps[0] = rgb2hsv(src.rgb);
	vec2 rad = TEXTURE_PIXEL_SIZE * 2.5;
	vec2 rad_broad = TEXTURE_PIXEL_SIZE * 7.0;
	for (int k = 0; k < 8; k++) {
		float a = float(k) * 0.7853981634;   // TAU / 8
		vec2 dirv = vec2(cos(a), sin(a));
		taps[k + 1] = rgb2hsv(texture(TEXTURE, UV + dirv * rad).rgb);
		// Broad ring staggered half a step so the two rings don't share spokes.
		float ab = a + 0.3926990817;         // TAU / 16
		taps[k + 9] = rgb2hsv(texture(TEXTURE, UV + vec2(cos(ab), sin(ab)) * rad_broad).rgb);
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
