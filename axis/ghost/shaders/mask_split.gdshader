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

// This layer's membership for the pixel: 1 well inside the hue target, feathered
// to 0 at the threshold edge, and 0 regardless for near-neutral pixels.
float chan_mask(float target_hue, vec3 hsv) {
	float d = hue_dist(hsv.x, target_hue);
	float m = 1.0 - smoothstep(u_threshold - u_feather, u_threshold + u_feather, d);
	return m * smoothstep(u_sat_floor, u_sat_floor + 0.08, hsv.y);
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

// Apply layer i to the running color. `w` is the layer's full on-screen weight
// (envelope x intensity x surface presence), already resolved CPU-side.
vec4 apply_layer(vec4 col, int i, vec3 hsv, vec2 uv) {
	float w = u_l_w[i];
	float mask = chan_mask(u_l_hue[i], hsv);
	if (mask * w <= 0.001) {
		return col;
	}
	int effect = u_l_effect[i];
	if (effect == 0) {
		// erase: fade this layer's pixels to black by weight. RGB only, alpha
		// untouched - this surface is drawn OVER the raw video, so a fading alpha
		// would let the raw color bleed back through the "erased" region.
		return vec4(col.rgb * (1.0 - w * mask), col.a);
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
	// accelerating hard at both ends so the slider genuinely REACHES fully-sparse
	// and fully-dense (the cubic hits 0 and 1 exactly; the threshold overshoots
	// the field's own range on both sides so nothing is consumed at 0 and
	// nothing survives at 1).
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
	vec3 fx = col.rgb * (1.0 - consume) + tint * rim * glow;
	return vec4(mix(col.rgb, fx, w * mask), col.a);
}

void fragment() {
	vec4 src = texture(TEXTURE, UV);
	vec3 hsv = rgb2hsv(src.rgb);
	vec4 col = src;
	for (int i = 0; i < MAXL; i++) {
		if (i >= u_l_count) {
			break;
		}
		col = apply_layer(col, i, hsv, UV);
	}
	COLOR = col;
}
