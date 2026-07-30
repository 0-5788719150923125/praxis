shader_type canvas_item;
// blend_disabled: like clown_paint, this WRITES A FIELD, not a picture - every
// channel is raw state, and the default blend_mix would multiply the colour
// channels by the guard alpha on write and mix in garbage from the
// never-cleared buffer.
render_mode blend_disabled;

// UMBRA's persistent shadow-mass field - the amorphous black body that lives
// inside the subject's own cast shadow. Runs in a ping-pong SubViewport pair
// (see MaskEditor._step_umbra_sim); mask_split.gdshader samples the result as
// u_umbra_field.
//
// Channels: R = mass density, G = rising essence (the smoke/flame trails),
// B = reserved (the planned facial transference - eyes/mouth deposits),
// A = the GUARD: where the mass is permitted to exist at all.
//
// Why a persistent field and not a drawn shape: the detector's region breathes
// ~15% frame to frame, and any shape drawn straight from it would pulse. The
// field only ever DEPOSITS into the region and then advects on its own, so
// detection wobble becomes a current rather than a twitch. Same school as the
// clown's liquid paint, and the same reason.
//
// The trails are the point of doing this as a field at all. The existing fire
// effect is scrolling noise, so nothing can ever detach from it; here the mass
// advects with an upward velocity that scales with its own density (buoyancy),
// so wisps genuinely tear off the top and rise, thinning as they climb.

uniform sampler2D u_prev : hint_default_black;
// The CPU detector's verdict, at grid resolution (see _update_umbra_model):
//   R = the LINKED shadow region (flood-grown from her silhouette, so every
//       cell in it is contiguous with her - that is the whole linkage test)
//   G = raw shadowness evidence (continuous, un-flooded)
//   B = the subject mask - her, the mic, and any other non-wall surface
uniform sampler2D u_region : hint_default_black;
uniform float u_dt = 0.016;        // playback seconds advanced this step
uniform int u_reset = 0;           // 1 = snap the field to its targets (seek/first frame)
uniform float u_time = 0.0;
uniform float u_aspect = 1.7778;
// The cast direction: from her centroid toward the shadow's, aspect-corrected
// and unit length. A near-STATIC property of the scene (the light does not
// move), so the editor EMAs it hard - re-deriving it per frame is exactly how
// the clown earned its uniform twitch.
uniform vec2 u_dir = vec2(1.0, 0.0);
uniform float u_loom = 0.45;       // Coverage: how far the mass grows outward along u_dir
uniform float u_rise = 1.0;        // Velocity: how fast the essence climbs
uniform float u_roil = 0.5;        // Contrast: turbulence in the currents
uniform float u_cling = 0.35;      // Cling: persistence in time
uniform float u_wisp = 0.0;        // Wisp: how readily essence detaches and rises
uniform float u_scale = 1.0;       // Scale: overall size of the mass

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

	// --- THE GUARD, and the looming.
	// The mass may exist in the linked shadow, plus a directional dilation of
	// it OUTWARD along the cast direction: walking back along -u_dir and taking
	// the strongest region cell found means growth can only ever bloom AWAY
	// from her, never toward her. The falloff with distance travelled is what
	// makes it thin as it looms rather than ending in a hard wall.
	float reach = (0.03 + 0.34 * clamp(u_loom, 0.0, 1.0)) * clamp(u_scale, 0.3, 2.5);
	// The region arrives at grid resolution, and sampling it straight gives the
	// mass a STAIRCASE edge - the detector's own cell grid, plainly visible in
	// the picture. Jittering every lookup by a little noise dissolves that
	// boundary into something organic; the CPU side also pre-softens the mask,
	// but this is what removes the last of the grid signature.
	vec2 jit = vec2(fbmu(uv * asp * 9.0 + vec2(3.1, u_time * 0.05)),
		fbmu(uv * asp * 9.0 + vec2(11.7, -u_time * 0.04))) - 0.5;
	jit *= 0.035;
	float allow = 0.0;
	for (int k = 0; k < 10; k++) {
		float f = float(k) / 9.0;
		vec2 s = uv + jit - (u_dir / asp) * (f * reach);
		float reg = texture(u_region, clamp(s, vec2(0.0), vec2(1.0))).r;
		// Gentle falloff: at 0.72 the loomed extension died away to almost
		// nothing and the mass read as a thin rind on the shadow's near edge
		// rather than a body looming out of it.
		allow = max(allow, reg * (1.0 - 0.45 * f * f));
	}
	vec3 here = texture(u_region, uv + jit).rgb;
	// NEVER on her. Independent of anything the simulation does downstream -
	// even if the field advects into her, this is a hard multiply at the source
	// and again at render time.
	// Read UNJITTERED. The jitter above exists to dissolve the region grid, but
	// displacing the exclusion lookup by the same offset would let the mass
	// creep a jitter's width onto her wherever it happens to point inward -
	// a few pixels, but on the one property this effect may never trade away.
	// The edge softening it costs is already paid for on the CPU side.
	float subj = texture(u_region, uv).b;
	allow *= 1.0 - smoothstep(0.20, 0.55, subj);
	allow = clamp(allow, 0.0, 1.0);

	// Deposit target: strongest where the footage itself shows real shadow,
	// still present (thinner) out in the loomed extension where there is no
	// evidence left to read - that outer band is the effect's own invention.
	// THE DEPOSIT MUST NOT BE A STATIC PLATE. Relaxing toward a fixed target
	// faster than the currents can move anything makes the field simply BE the
	// detector's region: measured, the mass changed no more between frames
	// than the static wall did. So the target itself boils, on a noise that
	// drifts upward - the mass is continuously churning even where the region
	// underneath it never moves.
	float churn = fbmu(uv * asp * 3.4 + vec2(u_time * 0.10, -u_time * 0.42));
	float tgt = allow * (0.52 + 0.48 * clamp(here.g, 0.0, 1.0)) * (0.50 + 1.00 * churn);

	// --- ADVECTION (semi-Lagrangian: sample where this parcel came FROM)
	vec4 prev0 = texture(u_prev, uv);
	vec2 cur = curlu(uv * asp * 2.6 + vec2(0.0, -u_time * 0.13));
	// buoyancy: denser mass climbs faster, which is what separates a body of
	// smoke from a uniformly drifting sheet
	vec2 vel = vec2(0.0, -1.0) * (0.030 + 0.110 * prev0.r) * clamp(u_rise, 0.05, 4.0)
		+ cur * (0.020 + 0.070 * clamp(u_roil, 0.0, 1.0))
		+ (u_dir / asp) * 0.020 * clamp(u_loom, 0.0, 1.0);
	vec4 prev = texture(u_prev, clamp(uv - vel * dt, vec2(0.0), vec2(1.0)));

	// --- MASS: decay what never reached a shadow, rise toward what did.
	// Rate-limited rather than latching to the target (the clown's v8 lesson:
	// an instant-max deposit with a slow decay makes the shape pulse).
	float cling = clamp(u_cling, 0.0, 1.0);
	float tau_dec = mix(0.22, 1.70, cling);
	// Slower than the clown's paint on purpose: paint wants to arrive fast and
	// stay put, but a body of smoke that snaps to its target has no currents
	// left to see. Cling trades liveliness for steadiness at this end.
	float tau_up = mix(0.10, 0.70, cling);
	float m = prev.r * exp(-dt / tau_dec);
	m += max(0.0, tgt - m) * (1.0 - exp(-dt / tau_up));
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
	// the essence is free of the guard's outward limit (it has left the
	// shadow), but it still may never drift onto her.
	w *= 1.0 - smoothstep(0.20, 0.55, subj);
	w = clamp(w, 0.0, 1.2);

	if (u_reset == 1) {
		m = tgt;
		w = 0.0;
	}
	// B stays 0: reserved for the facial transference pass (loose eye/mouth
	// voids deposited into the mass, led slightly ahead of her own motion).
	COLOR = vec4(m, w, 0.0, allow);
}
