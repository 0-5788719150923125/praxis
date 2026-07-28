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
// The live frame + the face's own colour statistics: the deposits below are
// EVIDENCE-DRIVEN, not stamped shapes. A landmark only says WHERE TO LOOK;
// the pixels say what the feature's shape is (dark socket, red lip line, lit
// nose ridge), the same way the white coat is a per-pixel face-tint match
// rather than a fitted oval. Stamped ellipses were the "circles drawn over
// the face" complaint.
uniform sampler2D u_frame : hint_default_black;
uniform float u_face_lum = 0.4;    // mean luminance over the detected face
uniform float u_face_red = 0.02;   // mean (r - g) over it - lips are redder than THIS
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

// A LOOK-HERE window: deliberately generous and very soft, so it bounds the
// search without ever drawing its own edge - whatever shape survives is the
// evidence's, not the window's.
float win(vec2 p, vec2 c, vec2 r) {
	return 1.0 - smoothstep(0.35, 1.0, length((p - c) / max(r, vec2(1e-4))));
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
	// EVIDENCE, read from the frame at this texel. Everything is relative to
	// the FACE's own statistics, never absolutes - the same discipline the
	// mouth detector uses, and the only thing that survives a monochrome or
	// heavily-graded clip.
	// DILATED evidence: the max over a small ring, not the single texel.
	// Raw per-pixel evidence traced a feature's outline (the lash line, the
	// lip line) but never covered it - the paint has to fill the eye, the
	// nose, the mouth. A max-filter grows each region outward from its own
	// strongest pixels, so coverage comes from the anatomy's real extent and
	// the contour stays organic (a floor would just redraw the disc).
	float dark = 0.0;
	float bright = 0.0;
	float redness = -1.0;
	for (int k = 0; k < 9; k++) {
		vec2 off = vec2(0.0);
		if (k > 0) {
			float a = float(k - 1) * 0.7853981634;
			off = vec2(cos(a), sin(a)) * 0.016;
		}
		vec3 c3 = texture(u_frame, clamp(UV + off / vec2(u_aspect, 1.0),
			vec2(0.0), vec2(1.0))).rgb;
		float l3 = dot(c3, vec3(0.299, 0.587, 0.114));
		dark = max(dark, clamp((u_face_lum - l3) / max(u_face_lum, 0.08), 0.0, 1.0));
		bright = max(bright, clamp((l3 - u_face_lum) / max(u_face_lum, 0.08), 0.0, 1.0));
		redness = max(redness, c3.r - c3.g);
	}
	float fred = redness;
	// EYES: the black covers what is actually DARK inside the look-window -
	// socket, lashes, lid crease, brow - so it wraps the eye's own shape
	// instead of capping a disc over it. Windows are per-eye and generous.
	float rl = clamp(u_eye_lr, 0.012, 0.08) * u_scale * 1.75;
	float rr = clamp(u_eye_rr, 0.012, 0.08) * u_scale * 1.75;
	float eye_ev = smoothstep(0.04, 0.26, dark);
	float wl = win(p, el, vec2(rl, rl * 1.15));
	float wr = win(p, er, vec2(rr, rr * 1.15));
	// The floor is small on purpose: it keeps a hint of paint where the
	// evidence is weak without ever filling the window as a shape.
	float tblack = max(wl, wr) * (0.22 + 0.78 * eye_ev);
	// LIPS: redder than the face's own mean, or the dark line of the mouth
	// itself - whichever is stronger. The look-window is wide and shallow
	// (a mouth's proportions), the evidence draws the lips.
	vec2 lr = vec2(clamp(u_mouth_r.x, 0.015, 0.055) * u_aspect * 1.9,
		clamp(u_mouth_r.y, 0.010, 0.032) * 1.7) * u_scale;
	float lip_ev = max(smoothstep(0.005, 0.075, fred - u_face_red),
		smoothstep(0.07, 0.38, dark) * 0.9);
	float tlip = win(p, mo, lr) * (0.18 + 0.82 * lip_ev);
	// NOSE: the one feature with no reliable 2D signature - a nose is a 3D
	// protrusion, so its evidence is the LIT RIDGE plus the shadow under the
	// tip. Window-led with a higher floor, brightness-shaped so the ball
	// still hugs the actual nose rather than sitting as a decal.
	float nr = clamp(rl * 0.55 + rr * 0.55, 0.018, 0.075) * u_scale;
	// Brightness leads. The shadow UNDER the nose is dark evidence too, and
	// leaning on it bridged the ball into the lips - so the dark term is a
	// weak, late-thresholded assist only, and the window stays compact.
	float nose_ev = max(smoothstep(0.0, 0.22, bright), smoothstep(0.22, 0.55, dark) * 0.45);
	float tnose = win(p, nc, vec2(nr, nr * 0.95)) * (0.42 + 0.58 * nose_ev);
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
