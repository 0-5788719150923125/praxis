shader_type canvas_item;
// blend_disabled: like clown_paint, this WRITES A FIELD, not a picture - every
// channel is raw state, and the default blend_mix would multiply the colour
// channels by the guard alpha on write and mix in garbage from the
// never-cleared buffer.
render_mode blend_disabled;

// UMBRA's persistent shadow-mass field - the amorphous black body of the ghost
// that stands beside her and moves her. Runs in a ping-pong SubViewport pair
// (see MaskEditor._step_umbra_sim); mask_split.gdshader samples the result as
// u_umbra_field.
//
// Channels: R = mass density, G = rising essence (the smoke/flame trails),
// B = reserved (the planned facial transference - eyes/mouth deposits),
// A = the GUARD: where the mass is permitted to exist at all.
//
// Why a persistent field and not a drawn shape: the silhouette arrives at 12 Hz
// on a 96x54 grid and it is a BITMAP, so a shape drawn straight from it would
// step at the sample rate and show its own cells. The field only ever DEPOSITS
// into the thrown silhouette and then advects on its own, so the track's
// discreteness becomes a current rather than a twitch. Same school as the
// clown's liquid paint, and the same reason.
//
// The trails are the point of doing this as a field at all. The existing fire
// effect is scrolling noise, so nothing can ever detach from it; here the mass
// advects with an upward velocity that scales with its own density (buoyancy),
// so wisps genuinely tear off the top and rise, thinning as they climb.

uniform sampler2D u_prev : hint_default_black;
// HER SILHOUETTE, at grid resolution, straight from the pose track (see
// MaskEditor._pt_upload_mask):
//   R = the person mask - the ghost's BODY, sampled through the cast transform
//   B = the same mask, dilated - the GUARD, sampled at plain screen uv
// Two channels of one mask at two dilations, on purpose: the body is her shape
// thrown somewhere else, and the guard is her shape where she actually is.
uniform sampler2D u_region : hint_default_black;
uniform float u_dt = 0.016;        // playback seconds advanced this step
uniform int u_reset = 0;           // 1 = snap the field to its targets (seek/first frame)
uniform float u_time = 0.0;
uniform float u_aspect = 1.7778;
// The cast direction, aspect-corrected and unit length: measured ONCE over the
// whole clip by the pose pre-pass (her shadow is the part of the background
// whose luminance varies as she moves) and biased upward by the editor before
// it gets here. A property of the ROOM, not of the frame - re-deriving it per
// frame is exactly how the clown earned its uniform twitch.
uniform vec2 u_dir = vec2(1.0, 0.0);
// THE CAST TRANSFORM, solved on the CPU (MaskEditor._umb_solve_cast) and handed
// over already inverted, because what a fragment asks is "what content LANDS
// here". It is a similarity pinned by two point pairs - her shoulder line to
// the ghost's anchor, her eye line to the ghost's head - so the ghost's head is
// wherever it was placed at any size, which a scale-about-a-pivot could never
// promise: magnification is also a WINDOW, and growing about any pivot near her
// walks the head off the top of the frame.
//
// All four are in ASPECT-CORRECTED space (x multiplied by u_aspect), because
// every one of them is a distance between two points on a body and raw uv is
// not a unit.
uniform vec2 u_inv0 = vec2(1.0, 0.0);        // inverse basis, column 0
uniform vec2 u_inv1 = vec2(0.0, 1.0);        // ...column 1
uniform vec2 u_anchor = vec2(0.9, 0.6);      // where her shoulder line lands
uniform vec2 u_src = vec2(0.9, 0.6);         // her own shoulder line
uniform vec2 u_eye_src = vec2(0.9, 0.2);     // her own eye line - the bust's centre
uniform float u_unit = 0.3;                  // her eye-to-shoulder distance
// THE BUST. Her silhouette is ~45% of a close-up frame; thrown whole at any
// magnification it is a wall of black with nothing in it to read (measured: 63%
// of the frame at Scale 2.8, and it read as a dimmed room rather than a
// figure). So the mass is her HEAD AND SHOULDERS - a radial falloff about her
// eye line, in HER OWN units, so it rides the transform with everything else -
// and below that the field's own smoke is what remains.
uniform float u_bust0 = 1.2;
uniform float u_bust1 = 2.6;
uniform float u_loom = 0.45;       // Coverage: how far down the body the bust reaches
uniform float u_rise = 1.0;        // Velocity: how fast the essence climbs
uniform float u_roil = 0.5;        // Contrast: turbulence in the currents
uniform float u_cling = 0.35;      // Cling: persistence in time
uniform float u_wisp = 0.0;        // Wisp: how readily essence detaches and rises
// THE GHOST'S EYES (channel B). HER eye landmarks, pushed through the same cast
// transform as the body (MaskEditor._umb_cast_from), so they are on the ghost's
// head by construction at any size - and they arrive already carrying the Lead,
// like everything else read from the track. Deposited into the field rather than
// drawn at render time so they inherit the same ooze and advection as the body:
// amorphous sockets that breathe, not two ellipses stuck on a mask.
uniform vec2 u_eye_l = vec2(0.5, 0.4);
uniform vec2 u_eye_r = vec2(0.6, 0.4);
uniform float u_eye_rad = 0.05;
uniform float u_eye_amt = 0.0;     // Gaze: 0 = no eyes at all

float permf(float x) {
	return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float hash21u(vec2 p) {
	p = p - floor(p / 289.0) * 289.0;
	return fract(permf(permf(p.x) + p.y) * 0.145598);
}

float vnoiseu(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float a = hash21u(i);
	float b = hash21u(i + vec2(1.0, 0.0));
	float c = hash21u(i + vec2(0.0, 1.0));
	float d = hash21u(i + vec2(1.0, 1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbmu(vec2 p) {
	float s = 0.0;
	float a = 0.5;
	for (int i = 0; i < 3; i++) {
		s += a * vnoiseu(p);
		p *= 2.03;
		a *= 0.5;
	}
	return s;
}

// Screen position -> the point of HER the ghost is showing there.
// Works in aspect-corrected space and returns the same, so the bust falloff can
// be measured in her own units without converting twice.
vec2 to_source(vec2 uva) {
	vec2 d = uva - u_anchor;
	return u_src + u_inv0 * d.x + u_inv1 * d.y;
}

// Sampled with explicit bounds: outside the frame there is simply no body, and
// letting the sampler CLAMP would smear the frame's edge row outward into a
// false silhouette wherever the ghost is thrown past the picture.
//
// But cutting it dead at the boundary is the other error, and it was visible: a
// ghost larger than she is samples past her own frame, so wherever it did, its
// body ended in a RULER-STRAIGHT line along the picture edge - a staircase in
// the corner that nothing else in the effect could explain. Fading out over a
// couple of cells is neither: no invented silhouette, no drawn edge.
float mask_at(vec2 p) {
	vec2 d = min(p, vec2(1.0) - p);          // distance to the nearest edge, per axis
	if (min(d.x, d.y) < -0.02) {
		return 0.0;
	}
	float fade = smoothstep(-0.02, 0.03, min(d.x, d.y));
	return texture(u_region, clamp(p, vec2(0.0), vec2(1.0)).xy).r * fade;
}

// Divergence-free-ish wander: the gradient of a scalar noise, rotated 90
// degrees. Curl rather than plain noise because a plain noise velocity field
// has sources and sinks - the mass visibly pools and drains at fixed spots,
// which reads as a texture rather than as smoke.
vec2 curlu(vec2 p) {
	float e = 0.09;
	float n1 = fbmu(p + vec2(0.0, e));
	float n2 = fbmu(p - vec2(0.0, e));
	float n3 = fbmu(p + vec2(e, 0.0));
	float n4 = fbmu(p - vec2(e, 0.0));
	return vec2(n1 - n2, n4 - n3) / (2.0 * e);
}

void fragment() {
	vec2 uv = UV;
	vec2 asp = vec2(u_aspect, 1.0);
	float dt = clamp(u_dt, 0.0, 0.1);

	// --- THE BODY, and the guard.
	// The mask arrives at grid resolution, and sampling it straight gives the
	// mass a STAIRCASE edge - the track's own cell grid, plainly visible in the
	// picture. Jittering every lookup by a little noise dissolves that boundary
	// into something organic.
	//
	// IT WAS DOING FAR MORE THAN THAT. At 0.035 this displaced the silhouette
	// lookup by +-34 px of a 1920-wide frame, ANIMATED, which is not a dissolved
	// grid edge - it is a crawling outline, and it was the whole of the reported
	// "very jittery and unstable" edge. The cell it exists to hide is now 12 px
	// rather than 20 (the track grid was doubled), so a third of the amplitude
	// covers it, and the drift is slowed to a creep so what remains reads as the
	// mass breathing rather than as the edge shaking.
	vec2 jit = vec2(fbmu(uv * asp * 9.0 + vec2(3.1, u_time * 0.017)),
		fbmu(uv * asp * 9.0 + vec2(11.7, -u_time * 0.013))) - 0.5;
	jit *= 0.010;
	// Everything about WHERE the ghost is happens in SOURCE space: the fragment
	// asks which point of her it is showing, and the whole body then moves,
	// turns and grows as one object. The version this replaces used the region
	// as a fixed permission stencil and slid a mass around underneath it, which
	// is why Pan appeared to move nothing but a few stray wisps.
	vec2 srca = to_source((uv + jit) * asp);
	float body = mask_at(vec2(srca.x / u_aspect, srca.y));
	// THE BUST. Radial, in her own units, about her eye line - so it survives
	// the transform unchanged and means the same thing at every Scale.
	float rr = length(srca - u_eye_src) / max(u_unit, 1e-4);
	// WHERE THE GHOST EXISTS. Note what is NOT in here: her own outline.
	float perm = clamp(body * (1.0 - smoothstep(u_bust0, u_bust1, rr)), 0.0, 1.0);
	// NEVER on her. A single grid-aligned lookup here draws her exclusion as a
	// clean geometric cutout - the track's own low-res cell boundary, and stark
	// against the organic, jittered edge everywhere else on the ghost, which is
	// what "huge bad ghost hand gap" was: a machined bite out of an otherwise
	// amorphous body wherever her real silhouette (an outstretched arm, reaching
	// for something the ghost's mass now also occupies) crosses it. MAX with a
	// tap at `jit` dissolves that edge the same way `body`'s is dissolved -
	// safely, because a maximum can only WIDEN the excluded area, never shrink
	// it, so the property this effect may never trade away (creeping onto her)
	// cannot regress; only the boundary's shape changes.
	float subj = max(texture(u_region, uv).b, texture(u_region, uv + jit).b);
	// THE GUARD GATES THE PICTURE, NOT THE SIMULATION - and that distinction is
	// the whole fix for "if a hand wipes through the ghost, the fill-in is
	// heavily delayed". Gating the DEPOSIT means every pixel her hand crosses is
	// emptied, and has to be rebuilt from zero at the rise time Cling sets once
	// the hand has gone - so the mass drags a hole behind every gesture. The mass
	// now keeps growing underneath her; only `allow` is cut, and the moment the
	// hand moves on the mass that was there all along is simply revealed. The
	// safety property is unchanged, because it was always a property of what is
	// DRAWN: mask_split multiplies everything it draws by this alpha.
	// The band starts LATER than it did (0.20 -> 0.34). At 0.20 the exclusion
	// began well outside her true outline - the mask's own soft rim plus a cell
	// of dilation - and left a strip of untouched wall between her and the mass.
	// Reported as a gap you could see the wall through.
	float allow = clamp(perm * (1.0 - smoothstep(0.34, 0.72, subj)), 0.0, 1.0);

	// THE DEPOSIT MUST NOT BE A STATIC PLATE. Relaxing toward a fixed target
	// faster than the currents can move anything makes the field simply BE its
	// input: measured on the version before this one, the mass changed no more
	// between frames than the static wall did. So the target itself boils, on a
	// noise that drifts upward - the ghost churns even while she holds still.
	float churn = fbmu(uv * asp * 3.4 + vec2(u_time * 0.10, -u_time * 0.42));
	// The churn MODULATES a solid deposit, it does not gate one. At 0.50+1.00x
	// it could drop the target to half and the resulting thin patches read as
	// holes in the ghost; the floor keeps the body continuous.
	float tgt = perm * (0.78 + 0.44 * churn);

	// --- ADVECTION (semi-Lagrangian: sample where this parcel came FROM)
	vec4 prev0 = texture(u_prev, uv);
	vec2 cur = curlu(uv * asp * 2.6 + vec2(0.0, -u_time * 0.13));
	// buoyancy: denser mass climbs faster, which is what separates a body of
	// smoke from a uniformly drifting sheet
	vec2 vel = vec2(0.0, -1.0) * (0.030 + 0.110 * prev0.r) * clamp(u_rise, 0.05, 4.0)
		+ cur * (0.020 + 0.070 * clamp(u_roil, 0.0, 1.0))
		+ (u_dir / asp) * 0.020 * clamp(u_loom, 0.0, 1.0);
	vec4 prev = texture(u_prev, clamp(uv - vel * dt, vec2(0.0), vec2(1.0)));

	// --- MASS: decay what is no longer inside the ghost, rise toward what is.
	// Rate-limited rather than latching to the target (the clown's v8 lesson:
	// an instant-max deposit with a slow decay makes the shape pulse).
	float cling = clamp(u_cling, 0.0, 1.0);
	float tau_dec = mix(0.22, 1.70, cling);
	// Slower than the clown's paint on purpose: paint wants to arrive fast and
	// stay put, but a body of smoke that snaps to its target has no currents
	// left to see. Cling trades liveliness for steadiness at this end.
	float tau_up = mix(0.10, 0.70, cling);
	float m = prev.r * exp(-dt / tau_dec);
	// A HOLE FILLS FASTER THAN THE MASS BREATHES. Cling's rise time is right for
	// the churn - the small, constant difference between the field and a target
	// that boils - and much too slow for a GAP, which is a large deficit that
	// wants to be gone at once. One rate cannot serve both: at the Cling the
	// author works at, the ghost's own fast movements (a head-shift, a hand a
	// step ahead of hers) opened a lagging void along their leading edge. Rate by
	// SIZE OF DEFICIT, so the two behaviours blend rather than trade.
	float gap = max(0.0, tgt - m);
	float urgency = smoothstep(0.12, 0.50, gap);
	m += gap * (1.0 - exp(-dt / mix(tau_up, 0.035, urgency)));
	m = clamp(m, 0.0, 1.4);

	// --- ESSENCE: fed from the mass, rising faster, dying sooner, and pinched
	// off in patches by a drifting noise so tendrils separate instead of the
	// whole top edge lifting as a sheet.
	vec2 velw = vec2(0.0, -1.0) * (0.105 + 0.26 * prev0.g) * clamp(u_rise, 0.05, 4.0)
		+ cur * (0.035 + 0.090 * clamp(u_roil, 0.0, 1.0));
	float wprev = texture(u_prev, clamp(uv - velw * dt, vec2(0.0), vec2(1.0))).g;
	// Essence leaves from the mass's UPPER SURFACE, the way smoke leaves a
	// fire - not uniformly out of its whole volume. `crest` is the vertical
	// gradient (mass here, nothing just above), so the feed concentrates
	// exactly along the top edge and the tendrils read as leaving something.
	float above = texture(u_prev, clamp(uv - vec2(0.0, 0.045), vec2(0.0), vec2(1.0))).r;
	float crest = clamp(m - above, 0.0, 1.0);
	// Threshold well inside fbm's actual range: 3 octaves sum to a mean near
	// 0.44, so the old 0.42..0.86 window was mostly returning zero and the
	// essence never reached a tenth of the mass (measured: peak 0.10).
	float pinch = smoothstep(0.30, 0.62, fbmu(uv * asp * 5.2 + vec2(u_time * 0.09, -u_time * 0.62)));
	float feed = clamp(u_wisp, 0.0, 1.0) * (0.30 * m + 2.60 * crest) * pinch * 7.0;
	float w = wprev * exp(-dt / mix(0.18, 0.95, cling));
	w += feed * dt;
	// The essence is likewise free to exist under her - `allow` is what stops it
	// being drawn there, and cutting it here would give the trails the same
	// refill lag the mass just lost.
	w = clamp(w, 0.0, 1.2);

	// --- EYES. Two soft sockets, roughened by the same noise family as the
	// body so their outline is ragged rather than elliptical, and only ever
	// present where there is mass to cut them out of - eyes floating on bare
	// wall would read as two spots, not as a face.
	float er = max(u_eye_rad, 0.004);
	float wob = fbmu(uv * asp * 7.0 + vec2(-u_time * 0.06, u_time * 0.11)) - 0.5;
	float dl = length((uv - u_eye_l) * asp) / er + wob * 0.55;
	float dr = length((uv - u_eye_r) * asp) / er + wob * 0.55;
	float socket = max(1.0 - smoothstep(0.35, 1.0, dl), 1.0 - smoothstep(0.35, 1.0, dr));
	float eyes_tgt = clamp(u_eye_amt, 0.0, 1.0) * socket * smoothstep(0.05, 0.30, m);
	float eprev = texture(u_prev, clamp(uv - vel * dt, vec2(0.0), vec2(1.0))).b;
	// Faster than the body: the eyes must track her, not drift with the smoke,
	// or the lead they were given is lost to the field's own inertia.
	float e = eprev * exp(-dt / 0.16);
	e += max(0.0, eyes_tgt - e) * (1.0 - exp(-dt / 0.07));
	e = clamp(e, 0.0, 1.0);

	if (u_reset == 1) {
		m = tgt;
		w = 0.0;
		e = eyes_tgt;
	}
	COLOR = vec4(m, w, e, allow);
}
