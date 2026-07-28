shader_type canvas_item;
// blend_disabled: this shader WRITES A FIELD, not a picture - all four
// channels are raw state. Under the default blend_mix the output RGB was
// multiplied by the coat's own alpha on write, and outside the face the
// never-cleared buffer's garbage alpha survived every step (the whole-frame
// coat bug).
render_mode blend_disabled;

// The clown's LIQUID PAINT field - a persistent, advected simulation the
// detector only DEPOSITS into. Runs in a ping-pong SubViewport pair (see
// MaskEditor._step_paint_sim); mask_split.gdshader samples the result as
// u_clown_paint instead of drawing feature shapes analytically.
//
// Channels: R = eye black (both patches), G = lip red, B = nose red,
// A = the white coat's face region (screen-space - the coat has NO fitted
// oval geometry left to rotate or rescale with anything else).
//
// Per step, per texel:
//   DEPOSIT - the current model's feature ellipses write in at full value
//             immediately (paint arrives fast where the feature IS);
//   ADVECT  - existing paint drifts toward its channel's attractor (the
//             feature's current position), so when a re-fit jumps, the
//             orphaned residue physically FLOWS into the new spot instead
//             of fading in place - the "colors morph into their new
//             position" behavior this exists for;
//   OOZE    - a slow noise wander + slight diffusion keeps the blobs
//             organic, liquid, never die-cut;
//   DECAY   - whatever never reaches a feature thins away (~0.5s).
//
// Like META's mirror, this is a bounded one-step-per-frame feedback loop -
// frame-rate coupled by nature, stepped on PLAYBACK time deltas so pause
// freezes the liquid and export traces the same currents.

uniform sampler2D u_prev : hint_default_black;
uniform float u_dt = 0.016;        // playback seconds advanced this step
uniform int u_reset = 0;           // 1 = snap the field to the targets (seek/first frame)
uniform float u_time = 0.0;        // playback clock, for the ooze wander
uniform float u_aspect = 1.7778;
uniform vec2 u_eye_l = vec2(0.42, 0.40);
uniform vec2 u_eye_r = vec2(0.58, 0.40);
uniform vec2 u_mouth = vec2(0.5, 0.62);
uniform vec2 u_nose = vec2(0.5, 0.52);       // its own slow anchor, not re-derived here
uniform vec2 u_face_c = vec2(0.5, 0.45);     // face centroid - the coat channel's home
uniform vec2 u_face_r = vec2(0.14, 0.20);    // face semi-axes (uv units)
// Feature sizes in PLAIN SCREEN units (uv height). Deliberately NOT
// eye-distance units: normalizing by the eye pair re-coupled every feature's
// size to the pair's jitter, and the whole mask rescaled in lockstep.
uniform float u_eye_lr = 0.035;
uniform float u_eye_rr = 0.035;
uniform vec2 u_mouth_r = vec2(0.05, 0.028);  // x in uv-x units, y in uv-y units
uniform float u_scale = 1.0;       // the clown layer's Scale knob (feature sizes)

float permf(float x) {
	return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float hash21p(vec2 p) {
	p = p - floor(p / 289.0) * 289.0;
	return fract(permf(permf(p.x) + p.y) * 0.145598);
}

float vnoisep(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float a = hash21p(i);
	float b = hash21p(i + vec2(1.0, 0.0));
	float c = hash21p(i + vec2(0.0, 1.0));
	float d = hash21p(i + vec2(1.0, 1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Soft ellipse: 1 well inside, feathered to 0 at the rim.
float ell(vec2 p, vec2 c, vec2 r) {
	return 1.0 - smoothstep(0.72, 1.0, length((p - c) / max(r, vec2(1e-4))));
}

// One channel's advected sample: paint drifts toward `attract` and wanders
// under the shared ooze field. Each channel advects toward ITS OWN
// attractor - the independence of the features lives here.
float advected(int ch, vec2 p, vec2 attract, vec2 asp, float dt) {
	vec2 wob = vec2(vnoisep(p * 5.0 + vec2(u_time * 0.5, 3.7)),
		vnoisep(p * 5.0 + vec2(19.1, -u_time * 0.4))) - 0.5;
	vec2 v = (attract - p) * 2.4 + wob * 0.35;
	vec2 src = clamp((p - v * dt) / asp, vec2(0.0), vec2(1.0));
	vec4 s = texture(u_prev, src);
	return ch == 0 ? s.r : (ch == 1 ? s.g : (ch == 2 ? s.b : s.a));
}

void fragment() {
	vec2 asp = vec2(u_aspect, 1.0);
	vec2 p = UV * asp;
	vec2 el = u_eye_l * asp;
	vec2 er = u_eye_r * asp;
	vec2 mo = u_mouth * asp;
	vec2 nc = u_nose * asp;
	vec2 fc = u_face_c * asp;
	// Targets: each feature's CURRENT ellipse, positioned at ITS detection,
	// sized by ITS OWN measured spread in screen units - no shared frame, no
	// shared scale, so nothing here can move or grow in lockstep.
	float rl = clamp(u_eye_lr, 0.012, 0.08) * u_scale;
	float rr = clamp(u_eye_rr, 0.012, 0.08) * u_scale;
	float tblack = max(ell(p, el, vec2(rl, rl * 1.3)), ell(p, er, vec2(rr, rr * 1.3)));
	// Lips are WIDE and shallow: the detected spread is the whole moving
	// mouth region (jaw and cheeks included), so x is stretched and y is
	// taken nearly raw - a square blob reads as a muzzle, not a mouth.
	vec2 lr = vec2(clamp(u_mouth_r.x, 0.015, 0.055) * u_aspect * 1.45,
		clamp(u_mouth_r.y, 0.010, 0.032) * 1.05) * u_scale;
	float tlip = ell(p, mo, lr);
	float nr = clamp(rl * 0.75 + rr * 0.75, 0.018, 0.07) * u_scale;
	float tnose = ell(p, nc, vec2(nr, nr * 1.08));
	// The coat's radii are CAPPED at a face-plausible size: on footage where
	// the key hue washes the whole room, the fitted spread is the ROOM's,
	// and an uncapped coat painted the wallpaper (again - the same lesson as
	// the old anchor-scale cap).
	vec2 fr = vec2(clamp(u_face_r.x, 0.05, 0.145) * u_aspect,
		clamp(u_face_r.y, 0.07, 0.205))
		* 1.05 * (0.75 + 0.25 * u_scale);
	float tface = 1.0 - smoothstep(0.72, 1.05, length((p - fc) / fr));
	vec4 tgt = vec4(tblack, tlip, tnose, tface);
	if (u_reset == 1) {
		COLOR = tgt;
	} else {
		// Carry: advection toward each channel's OWN attractor (the eye
		// channel flows to whichever eye is nearer, the coat to the face
		// centre), folded with a touch of blur so edges soften in transit.
		vec2 eatt = distance(p, el) < distance(p, er) ? el : er;
		vec4 adv = vec4(
			advected(0, p, eatt, asp, u_dt),
			advected(1, p, mo, asp, u_dt),
			advected(2, p, nc, asp, u_dt),
			advected(3, p, fc, asp, u_dt));
		vec2 px = 1.0 / vec2(textureSize(u_prev, 0));
		vec4 blur = (texture(u_prev, UV + vec2(px.x, 0.0))
			+ texture(u_prev, UV - vec2(px.x, 0.0))
			+ texture(u_prev, UV + vec2(0.0, px.y))
			+ texture(u_prev, UV - vec2(0.0, px.y))) * 0.25;
		vec4 carried = mix(adv, blur, 0.18);
		// Decay what never arrives; deposit is instant and dominant.
		float dec = exp(-u_dt / 0.5);
		vec4 f = max(carried * dec - vec4(0.004 * u_dt * 60.0) * 0.02, vec4(0.0));
		COLOR = max(f, tgt);
	}
}
