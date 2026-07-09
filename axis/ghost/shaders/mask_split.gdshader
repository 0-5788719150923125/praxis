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
uniform float u_l_speed[6];        // drift velocity multiplier (the Velocity slider)
uniform float u_l_smooth[6];       // echo's temporal blend is baked into u_l_ew CPU-side;
                                    // this is the raw fx_smooth value, for effects (snow's
                                    // Gust) that want it directly

uniform float u_time = 0.0;        // clip playback seconds (deterministic clock)
uniform float u_aspect = 1.7778;   // video w/h, so the pattern stays isotropic

// The temporal echo ring: quarter-res snapshots of PAST frames, captured by the
// editor every fixed slice of PLAYBACK time (not wall clock) and pushed here in
// AGE ORDER - u_echo0 is the newest capture, u_echo3 the oldest. gdshader has no
// sampler arrays, hence four uniforms and the if-chain in echo_frame().
uniform sampler2D u_echo0 : hint_default_black;
uniform sampler2D u_echo1 : hint_default_black;
uniform sampler2D u_echo2 : hint_default_black;
uniform sampler2D u_echo3 : hint_default_black;
uniform sampler2D u_echo4 : hint_default_black;
uniform sampler2D u_echo5 : hint_default_black;
uniform sampler2D u_echo6 : hint_default_black;
uniform sampler2D u_echo7 : hint_default_black;

// Per-layer temporal kernel over the ring (6 layers x 8 ages, normalized,
// recomputed CPU-side every frame): how the echo's base coat blends the past.
// A single 1.0 = discrete stutter at the chosen lag; a spread = the Smoothing
// slider's EMA-like temporal blend, continuously shifting with the playhead so
// nothing steps. u_l_elag is each layer's integer lag, the clones' start age.
uniform float u_l_ew[48];
uniform int u_l_elag[6];

// The whisp anchor: the target color's mass centroid in the frame, EMA-tracked
// CPU-side over short windows (see MaskEditor) - the whisp field's coordinate
// origin rides it, so the pattern auto-locks to the landmark (a lit face half)
// and follows it smoothly as it moves.
uniform vec2 u_anchor = vec2(0.5, 0.5);

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

vec2 hash22(vec2 p) {
	p = p - floor(p / 289.0) * 289.0;
	float a = permf(permf(p.x) + p.y);
	return vec2(fract(a * 0.145598), fract(permf(a) * 0.145598));
}

// Two nearest voronoi feature distances + the winning cell's id, for the crystal
// facets: F2-F1 is small exactly along facet borders (the bright edge line), and
// the cell id keys each facet's brightness and refraction offset.
vec3 voronoi_f1f2(vec2 p) {
	vec2 ip = floor(p);
	vec2 fp = fract(p);
	float f1 = 8.0;
	float f2 = 8.0;
	float id = 0.0;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec2 g = vec2(float(x), float(y));
			vec2 o = hash22(ip + g);
			float d = length(g + o - fp);
			if (d < f1) {
				f2 = f1;
				f1 = d;
				id = hash21(ip + g);
			} else if (d < f2) {
				f2 = d;
			}
		}
	}
	return vec3(f1, f2, id);
}

// One past frame from the echo ring by age (0 = newest capture .. 7 = oldest).
vec3 echo_frame(int age, vec2 uv) {
	if (age <= 0) { return texture(u_echo0, uv).rgb; }
	if (age == 1) { return texture(u_echo1, uv).rgb; }
	if (age == 2) { return texture(u_echo2, uv).rgb; }
	if (age == 3) { return texture(u_echo3, uv).rgb; }
	if (age == 4) { return texture(u_echo4, uv).rgb; }
	if (age == 5) { return texture(u_echo5, uv).rgb; }
	if (age == 6) { return texture(u_echo6, uv).rgb; }
	return texture(u_echo7, uv).rgb;
}

// One octave of falling snow: unit cells, a sparse subset of them (gated by
// `thr`) each carrying one soft round flake at a per-cell random offset and
// radius. Three octaves of this at different scale/fall-speed/threshold (see
// the snow branch in apply_layer) give the near/mid/far spread a single
// octave can't - one size and spacing reads as a grid, not weather.
float snow_layer(vec2 p, float thr) {
	vec2 cell = floor(p);
	vec2 f = fract(p);
	vec2 rnd = hash22(cell);
	if (rnd.x > thr) {
		return 0.0;
	}
	vec2 center = vec2(0.5) + (hash22(cell + 91.7) - 0.5) * 0.7;
	float r = 0.10 + rnd.y * 0.16;
	float d = length(f - center);
	return 1.0 - smoothstep(r * 0.25, r, d);
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
	}
	// (freeze and whisp are evaluated inline in apply_layer - both consume the
	// tap ring's gradient probe, which this function can't see.)
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
vec4 apply_layer(vec4 col, int i, vec3 taps[25], vec2 uv, sampler2D tex) {
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
	if (effect == 6) {
		// CRYSTAL - fractal faceted glass IN PLACE of the target color. Same
		// projection foundation as erase (linear, gate-free, ring-proof): p is
		// "how much of this pixel is the target color", and exactly that much
		// of the pixel becomes crystal. Two voronoi octaves make the fractal
		// facets (the second subdivides the first); each facet refracts the
		// footage through its own fixed offset and carries its own brightness,
		// with cold edge light along facet borders (fixed palette - never the
		// key hue). Contrast sharpens/brightens the borders; scale sizes the
		// facets; pan slides the lattice.
		float k6 = shade_amount(w, uv);
		float lum6 = dot(col.rgb, vec3(0.299, 0.587, 0.114));
		vec3 ch6 = col.rgb - vec3(lum6);
		float p6 = max(0.0, dot(ch6, u_l_tdir[i]));
		// REPLACEMENT AMOUNT IS THE ALIGNED FRACTION, not the raw projection.
		// The projection measures ABSOLUTE target chroma - tiny on dim footage
		// even when the pixel is purely the target color, so a replacement
		// mixed by it barely registers (erase gets away with raw p because
		// subtracting a quarter of a color visibly desaturates; replacing a
		// quarter of a dark pixel with a ghost shows nothing). frac = how
		// aligned the pixel's own color is with the target (brightness-free),
		// smoothly ramped by whether the pixel has any chroma to speak of.
		float frac6 = p6 / max(length(ch6), 1e-4);
		float amt6 = clamp(frac6 * smoothstep(0.02, 0.10, length(ch6)) * k6 * 1.2, 0.0, 1.0);
		if (amt6 <= 0.003) {
			return col;
		}
		vec2 cp = (uv * vec2(u_aspect, 1.0) + u_l_off[i]) * (7.0 * u_l_scale[i]);
		vec3 v1 = voronoi_f1f2(cp);
		vec3 v2 = voronoi_f1f2(cp * 2.7 + 13.7);
		float con6 = u_l_con[i];
		float edge_w = 0.03 + 0.09 * (1.0 - con6);
		float edge1 = 1.0 - smoothstep(0.0, edge_w, v1.y - v1.x);
		float edge2 = 1.0 - smoothstep(0.0, edge_w * 1.8, v2.y - v2.x);
		vec2 fo = (hash22(vec2(v1.z, v2.z) * 289.0) - 0.5) * 0.05;
		vec3 glass = texture(tex, clamp(uv + fo, vec2(0.0), vec2(1.0))).rgb;
		float facet = 0.45 + 0.55 * v1.z;
		vec3 cryst = glass * facet * vec3(0.78, 0.90, 1.08);
		cryst += vec3(0.72, 0.86, 1.0) * (edge1 * 0.55 + edge2 * 0.20) * (0.4 + 0.6 * con6);
		return vec4(mix(col.rgb, cryst, amt6), col.a);
	}
	if (effect == 7) {
		// ECHO - temporal lag. The target color's region stops being live: it
		// shows a muted (desaturated, dimmed) PAST frame from the echo ring
		// instead, so one side of the scene stays fully real and continuous
		// while the other becomes a delayed, stuttering ghost of itself.
		// Coverage adds CLONES - additional, older echoes - each shifted by an
		// accumulating pan*scale offset and painted oldest-first, stacking the
		// ages into staircase time-shapes. Contrast is how muted the ghosts
		// are. Same projection foundation as erase: no gates, no rings.
		float k7 = shade_amount(w, uv);
		float lum7 = dot(col.rgb, vec3(0.299, 0.587, 0.114));
		vec3 ch7 = col.rgb - vec3(lum7);
		float p7 = max(0.0, dot(ch7, u_l_tdir[i]));
		// Aligned fraction, not raw projection - see the crystal branch.
		float frac7 = p7 / max(length(ch7), 1e-4);
		float amt7 = clamp(frac7 * smoothstep(0.02, 0.10, length(ch7)) * k7 * 1.2, 0.0, 1.0);
		if (amt7 <= 0.003) {
			return col;
		}
		int clones = 1 + int(round(clamp(u_l_dens[i], 0.0, 1.0) * 3.0));
		vec2 step_uv = u_l_off[i] * 0.12 * u_l_scale[i];
		float mute = 0.35 + 0.5 * u_l_con[i];
		// BASE FIRST: the unshifted lagged coat replaces the keyed region
		// (painted last at full alpha it would wipe the displaced clones - the
		// first cut drew an empty staircase). It blends the ring through the
		// layer's temporal kernel (u_l_ew): a lone weight = held-frame stutter
		// at the chosen lag, a spread = the Smoothing slider's continuous
		// temporal smear. The offset clones then layer over it, starting at
		// the base lag and stepping deeper into the past.
		vec3 base = vec3(0.0);
		for (int k = 0; k < 8; k++) {
			base += echo_frame(k, uv) * u_l_ew[i * 8 + k];
		}
		float bl = dot(base, vec3(0.299, 0.587, 0.114));
		base = mix(base, vec3(bl), mute) * 0.9;
		vec3 acc = mix(col.rgb, base, amt7);
		for (int c = 1; c < 4; c++) {
			if (c >= clones) {
				break;
			}
			vec2 euv = uv - step_uv * float(c);
			// A clone whose shifted sample leaves the frame must FADE out at
			// the border, not stop - the hard skip cut a screen-space
			// rectangle into the composite (the "hard black box" artifact).
			float ef = smoothstep(0.0, 0.06, euv.x) * smoothstep(0.0, 0.06, 1.0 - euv.x)
				* smoothstep(0.0, 0.06, euv.y) * smoothstep(0.0, 0.06, 1.0 - euv.y);
			if (ef <= 0.001) {
				continue;
			}
			vec3 e = echo_frame(min(u_l_elag[i] + c, 7), clamp(euv, vec2(0.0), vec2(1.0)));
			float el = dot(e, vec3(0.299, 0.587, 0.114));
			e = mix(e, vec3(el), mute) * (0.85 - 0.13 * float(c));
			acc = mix(acc, e, amt7 * (0.55 - 0.09 * float(c)) * ef);
		}
		return vec4(acc, col.a);
	}
	if (effect == 9) {
		// SNOW - the one effect with no key color at all: hue_a is unused, so
		// there's nothing here for tap_mask/mask_over to classify against.
		// Foreground vs. background is decided per pixel, automatically, from
		// MOTION (the echo ring's two newest captures, ~0.35s apart - the same
		// temporal capture whisp/echo already drive) and COLOR INTENSITY
		// (chroma magnitude): a lit, moving subject scores high on both, a
		// static, desaturated background scores low on both. Feathered, never
		// a hard mask - the flakes thin out approaching the subject instead of
		// stepping across a line. Sensitivity (Contrast, relabeled in the
		// editor) slides where that cut falls.
		vec3 e0 = echo_frame(0, uv);
		vec3 e1 = echo_frame(1, uv);
		float motion9 = abs(dot(e0 - e1, vec3(0.299, 0.587, 0.114)));
		float lum9 = dot(col.rgb, vec3(0.299, 0.587, 0.114));
		float chroma9 = length(col.rgb - vec3(lum9));
		float fg9 = clamp(motion9 * 3.5 + chroma9 * 1.3, 0.0, 1.0);
		float sens9 = u_l_con[i];
		// Per-size-class edge, not one shared cut. Large/NEAR flakes should read
		// as stable, solid shapes - a big disc getting a jittery bite taken out
		// of it right at the (coarse, ~3Hz) motion probe's noisiest zone reads as
		// an error, not weather, so they hold out for strong, sustained evidence
		// (a high threshold) and fade on a WIDE margin (no hard edge to bite).
		// Small/FAR flakes are the atmospheric ones - flickering in and out near
		// a moving, colorful subject IS the distance cue, so they fold at the
		// first hint (a low threshold, a narrow margin).
		float edge_small = mix(0.12, 0.55, sens9);
		float edge_big = mix(0.22, 0.85, sens9);
		float bg_small = 1.0 - smoothstep(edge_small - 0.09, edge_small + 0.09, fg9);
		float bg_mid = 1.0 - smoothstep(mix(edge_small, edge_big, 0.5) - 0.11,
			mix(edge_small, edge_big, 0.5) + 0.11, fg9);
		float bg_big = 1.0 - smoothstep(edge_big - 0.16, edge_big + 0.16, fg9);
		if (max(bg_small, bg_big) * w <= 0.003) {
			return col;
		}
		// No static pan for snow - there's no key-color landmark to place a
		// pattern on, only a direction to fall in - so Pan X/Y (u_l_off,
		// relabeled Wind X/Y in the editor) goes entirely into that direction
		// instead of into the sample coordinate.
		vec2 p9 = (uv * vec2(u_aspect, 1.0)) * (16.0 * u_l_scale[i]);
		float dens9 = clamp(u_l_dens[i], 0.0, 1.0);
		float thr9 = mix(0.03, 0.6, dens9);
		float t9 = u_time * u_l_speed[i];
		// Fall direction: straight down by default (Wind X/Y at 0,0 keeps it
		// plumb), steered by the wind vector otherwise. Kept a pure direction
		// (normalized) - speed stays on the Velocity slider, not coupled to how
		// hard the wind blows sideways.
		vec2 raw_dir = vec2(u_l_off[i].x, 1.0 + u_l_off[i].y);
		vec2 fall_dir = length(raw_dir) > 0.001 ? normalize(raw_dir) : vec2(0.0, 1.0);
		// Gust (the Gust slider, fx_smooth - unused by echo's own smoothing math
		// for this effect, so free to repurpose): a slow, irregular field swings
		// the local direction and speed together rather than smoothly - real
		// gusts are bursty, not sinusoidal. 0 = today's perfectly even drift; 1 =
		// swings most of the way around and surges the speed.
		float gust9 = clamp(u_l_smooth[i], 0.0, 1.0);
		vec2 gust_p = uv * 0.7 + vec2(t9 * 0.045, t9 * 0.03);
		float gust_ang = (fbm(gust_p) - 0.5) * 6.2832 * gust9;
		float gust_spd = 1.0 + (fbm(gust_p + 11.3) - 0.5) * 1.6 * gust9;
		float gca = cos(gust_ang);
		float gsa = sin(gust_ang);
		vec2 gusted_dir = vec2(gca * fall_dir.x - gsa * fall_dir.y, gsa * fall_dir.x + gca * fall_dir.y);
		// SUBTRACTED, not added: sampling further along the fall direction as
		// time advances is what makes the PATTERN appear to move that way - a
		// coordinate offset added to a sample grows the sample position, which
		// visually scrolls the content the OPPOSITE way (this is what had snow
		// falling up instead of down).
		vec2 drift9 = gusted_dir * t9 * gust_spd;
		// Three depth octaves, weighted toward the small/far end - a field is
		// mostly fine, distant flecks with a few larger near ones as accents,
		// not the other way around (a handful of big shapes reading as the
		// whole effect crowds out everything else). Cell scale sets flake SIZE
		// (fewer, bigger cells at 0.55x - up to more, smaller cells at 2.3x), and
		// each octave's own bg_* gate is chosen to match: the big/near octave
		// gets the stable gate, the small/far one gets the flickery gate. Each
		// also rides the shared gust field at its own rate (0.45/0.85/1.5x), so
		// big/near and small/far flakes drift apart under gust instead of
		// moving as one rigid sheet.
		float big = snow_layer(p9 * 0.55 + vec2(0.0, 0.0) - drift9 * 0.45, thr9 * 0.22) * 0.4 * bg_big;
		float mid = snow_layer(p9 * 1.15 + vec2(-0.4, 0.3) - drift9 * 0.85, thr9 * 0.55) * 0.65 * bg_mid;
		float small = snow_layer(p9 * 2.3 + vec2(0.7, -0.5) - drift9 * 1.5, thr9) * 1.0 * bg_small;
		float flake = clamp(big + mid + small, 0.0, 1.0);
		vec3 snowc = vec3(0.88, 0.93, 1.0) * clamp(u_l_glow[i], 0.6, 1.6);
		// bg_small/bg_mid/bg_big are already folded into big/mid/small above -
		// flake doesn't need a second gate here, only the layer's own weight.
		float amt9 = clamp(flake * w, 0.0, 1.0);
		return vec4(mix(col.rgb, snowc, amt9), col.a);
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
	float field;
	float t_eff = u_time * u_l_speed[i];
	if (effect == 5) {
		// WHISP - the content-aware volumetric. Its field is not independent of
		// the picture: the fine tap ring doubles as a luminance-gradient probe,
		// and the field's coordinates are advected ALONG the features (the
		// gradient's tangent), so tendrils curl around edges of the face rather
		// than drifting through them. The coordinate origin rides u_anchor -
		// the target color's mass centroid, EMA-tracked CPU-side - so the whole
		// pattern auto-locks to the landmark and follows it as it moves.
		// The gradient probe reads the MID and WIDE rings, not the fine one:
		// 2.5-texel luminance differences shimmer frame-to-frame with
		// compression noise (whisp was the one volumetric that jittered), while
		// the 7/22-texel rings see the actual features - spatially smoother and
		// temporally stable, and feature-following is about features, not grain.
		vec2 grad = vec2(0.0);
		for (int k2 = 0; k2 < 8; k2++) {
			float am = float(k2) * 0.7853981634 + 0.3926990817;
			float aw = float(k2) * 0.7853981634 + 0.1963495409;
			grad += vec2(cos(am), sin(am)) * taps[k2 + 9].z
				+ vec2(cos(aw), sin(aw)) * taps[k2 + 17].z;
		}
		vec2 flow = vec2(-grad.y, grad.x) * 1.5;
		// LOCAL anchor pull, not rigid translation: full tracking at the
		// landmark, exponentially fading with distance from it - the field
		// near the face rides the lock while the far field stays put. Applied
		// globally, every anchor correction translated the ENTIRE pattern as
		// one body ("the whole field bounces in random directions").
		float near5 = exp(-3.0 * length(uv - u_anchor));
		vec2 p5 = ((uv - (u_anchor - vec2(0.5)) * near5) * vec2(u_aspect, 1.0) + u_l_off[i])
			* (3.0 * u_l_scale[i]);
		vec2 drift5 = vec2(0.22 * t_eff, -0.12 * t_eff);
		vec2 wv = vec2(fbm(p5 * 1.2 + drift5 * 0.6), fbm(p5 * 1.4 - drift5 * 0.4));
		field = fbm(p5 + flow + drift5 + 0.9 * wv);
	} else if (effect == 2) {
		// FREEZE with sparse content conformance: the same mid/wide-ring
		// gradient probe as whisp, but here its MAGNITUDE gates a tangent
		// warp - where the picture concentrates structure (the face), the
		// veins bend along the features like line-work tattooed onto it;
		// across flat regions the probe reads ~0 and the veins float exactly
		// as before. Sticking is local and earned by the pixels, never global.
		vec2 grad2 = vec2(0.0);
		for (int k3 = 0; k3 < 8; k3++) {
			float am2 = float(k3) * 0.7853981634 + 0.3926990817;
			float aw2 = float(k3) * 0.7853981634 + 0.1963495409;
			grad2 += vec2(cos(am2), sin(am2)) * taps[k3 + 9].z
				+ vec2(cos(aw2), sin(aw2)) * taps[k3 + 17].z;
		}
		float feat = smoothstep(0.06, 0.30, length(grad2));
		vec2 conform = vec2(-grad2.y, grad2.x) * 1.4 * feat;
		vec2 p2 = (uv * vec2(u_aspect, 1.0) + u_l_off[i]) * (3.0 * u_l_scale[i]);
		vec2 drift2 = vec2(0.02 * t_eff, 0.015 * t_eff);
		float n2 = fbm(p2 * 1.4 + drift2 + conform);
		field = pow(1.0 - abs(2.0 * n2 - 1.0), 2.2);
	} else {
		field = wisp_field(uv, effect, t_eff, u_l_off[i], u_l_scale[i]);
	}
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
	float shape = (effect == 2 || effect == 5) ? consume : max(consume, lip);
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
	} else if (effect == 5) {
		// whisp: soft pearl streamers - a spectral, moon-grey light distinct
		// from fire's amber and freeze's ice blue.
		emiss = vec3(0.72, 0.82, 0.92) * (heat * 0.45 + rim * 0.5);
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
		col = apply_layer(col, i, taps, UV, TEXTURE);
	}
	COLOR = col;
}
