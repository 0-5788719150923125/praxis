shader_type canvas_item;

// Mask split - chroma-key BY COLOR, not by screen position.
//
// The source footage is lit with distinct color regions (e.g. a hard red/blue
// key-light split on a face) - the boundary moves in screen space as the subject
// moves, so this can't be a spatial crop. Every pixel is scored per CHANNEL: each
// channel targets one reference hue, and a pixel's membership is its hue-distance
// to that target (inside threshold = full member, feather = soft edge), gated by a
// saturation floor so near-neutral pixels (shadow, highlight, background) never key.
//
// Channels are INDEPENDENT, not competitive: each computes its own mask and applies
// its own effect, in order - so one channel (the default working set) can erase the
// reds while a second, when enabled, treats the blues differently, and their masks
// can overlap and layer. This replaced an earlier A-vs-B classifier (every pixel
// forced to one side or the other), which made a single-color target awkward and
// couldn't layer.
//
// Per-channel effects (see MaskSession.MASK_EFFECTS): "erase" is real; "fire"/
// "freeze" are placeholder tints until their real fields (heat, frame-history)
// land - the uniforms they'll need are already wired so swapping the placeholder
// for the real field later doesn't touch the channel logic above it.

uniform float u_hue_a = 0.02;      // channel 1 target hue, 0..1
uniform float u_hue_b = 0.58;      // channel 2 target hue, 0..1
uniform float u_threshold = 0.24;  // hue-distance inside which a pixel fully belongs
uniform float u_feather = 0.12;    // softness of the membership edge
uniform float u_sat_floor = 0.18;  // below this saturation, a pixel keys to NO channel

uniform int u_effect_a = 0;        // index into MaskSession.MASK_EFFECTS
uniform int u_effect_b = 0;
// Already ramp/decay-shaped by MaskSession.at_time() before it reaches here - a
// marker's own ramp/decay span eases these directly, so the shader itself doesn't
// need to know a transition is even happening. intensity 0 = channel off entirely.
uniform float u_intensity_a = 1.0;
uniform float u_intensity_b = 0.0;

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

// This channel's membership for the pixel: 1 well inside the hue target, feathered
// to 0 at the threshold edge, and 0 regardless for near-neutral pixels.
float chan_mask(float target_hue, vec3 hsv) {
	float d = hue_dist(hsv.x, target_hue);
	float m = 1.0 - smoothstep(u_threshold - u_feather, u_threshold + u_feather, d);
	return m * smoothstep(u_sat_floor, u_sat_floor + 0.08, hsv.y);
}

// Placeholder per-channel treatments (see header): "erase" is the real, requested
// effect; the other two are honest stand-ins, not the finished fields.
vec4 apply_effect(vec4 col, int effect, float intensity, float mask) {
	if (effect == 1) {          // fire (placeholder: warm boost, real field is TODO)
		vec3 warm = col.rgb + vec3(0.35, 0.12, -0.15) * intensity;
		return vec4(mix(col.rgb, warm, mask), col.a);
	} else if (effect == 2) {   // freeze (placeholder: cool desaturate, real field is TODO)
		float g = dot(col.rgb, vec3(0.299, 0.587, 0.114));
		vec3 cold = mix(col.rgb, vec3(g * 0.85, g * 0.95, g * 1.15), 0.6);
		return vec4(mix(col.rgb, cold, mask), col.a);
	}
	// erase: fade this channel's pixels to black by intensity. RGB only, alpha
	// untouched - this layer is drawn OVER the raw video (see mask_editor.gd's
	// composition), so a fading alpha would let the raw color underneath bleed back
	// through exactly the region that's supposed to read as erased.
	return vec4(col.rgb * (1.0 - intensity * mask), col.a);
}

void fragment() {
	vec4 src = texture(TEXTURE, UV);
	vec3 hsv = rgb2hsv(src.rgb);
	vec4 col = src;
	col = apply_effect(col, u_effect_a, u_intensity_a, chan_mask(u_hue_a, hsv));
	col = apply_effect(col, u_effect_b, u_intensity_b, chan_mask(u_hue_b, hsv));
	COLOR = col;
}
