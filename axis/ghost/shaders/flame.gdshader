shader_type canvas_item;
render_mode blend_add;

// Flame field - the volumetric body of Layer.Fire.
//
// The quad covers the frame; every pixel evaluates one shared TEMPERATURE FIELD:
// a row of heat SOURCES along the fire bed (each one driven by a harmonic band on
// the CPU side) raises a column of heat, and rising domain-warped fbm noise CARVES
// that column - the subtraction cuts tongues into the upper edge and pinches some
// free entirely, so licks form, detach, climb and dissolve on their own. That is
// where the life comes from: the flame is one field, not a crowd of particles.
//
// Time comes in as a uniform (the layer's own clock), never TIME - live and the
// offline Movie Maker bake step the same clock, so a seed reproduces its fire.

// The fire bed sources (unit-fraction x, heat 0..~1.2, gaussian width).
uniform int u_n = 6;
uniform float u_x[8];
uniform float u_heat[8];
uniform float u_w[8];

uniform float u_time = 0.0;
uniform vec2 u_half = vec2(0.9, 0.55); // quad half-extents in unit fractions
uniform float u_base = 0.52;           // bed y (unit fractions, +down)
uniform float u_burst = 0.0;           // beat surge: taller, wilder, brighter
uniform float u_wind = 0.0;            // sideways shear, grows with height
uniform float u_scale = 1.0;           // noise scale        (sampled per instance)
uniform float u_turb = 0.7;            // domain-warp gain   (sampled per instance)
uniform float u_lift = 1.0;            // column height gain (sampled per instance)
uniform float u_gain = 1.0;            // brightness gain    (sampled per instance)
uniform vec2 u_seed = vec2(0.0);       // noise offset       (sampled per instance)

// Lattice hash on a mod-289 permutation polynomial: inputs are bounded BEFORE any
// multiply, so precision holds no matter how far the time scroll pushes the
// coordinates. (A fract-of-big-product hash quantizes within seconds of scrolling -
// the upper fbm octaves blow coordinates past float precision and the flame
// interior tiles into blocks.) The noise repeats every 289 lattice units: at these
// scales that is minutes of scroll, invisible.
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
	float a = 0.5;
	for (int i = 0; i < 5; i++) {
		v += a * vnoise(p);
		p = mat2(vec2(1.6, 1.2), vec2(-1.2, 1.6)) * p;
		a *= 0.5;
	}
	return v;
}

void fragment() {
	vec4 tint = COLOR; // white quad texture x modulate: the scene's fade rides in here

	vec2 p = (UV - 0.5) * 2.0 * u_half; // unit fractions, y down, (0,0) at centre
	float yu = u_base - p.y;            // height above the bed, up-positive
	float y = max(yu, 0.0);

	// The heat envelope: each source is a gaussian column that leans with the wind
	// as it rises. env at the BED is what the harmonics put in.
	float env = 0.0;
	for (int i = 0; i < u_n; i++) {
		float d = (p.x - u_x[i] - u_wind * y * 0.7) / u_w[i];
		env += u_heat[i] * exp(-d * d);
	}
	env = min(env, 1.25);

	// Column height budget: strongly nonlinear in the heat, so quiet passages sit
	// as embers hugging the bed and powerful ones send columns most of the screen.
	float H = 0.07 + 1.10 * u_lift * pow(max(env, 0.0), 1.25) * (1.0 + 0.9 * u_burst);
	float v = y / max(H, 1e-3); // 0 at the bed .. 1 at the nominal tip

	// Rising, domain-warped turbulence. The sample point scrolls DOWN in noise
	// space, so the features CLIMB; the warp grows with height and with the burst,
	// so tips shred harder than roots.
	vec2 q = vec2(p.x * 2.4, yu * 1.7) * u_scale + u_seed;
	q.y += u_time * 1.25;
	q.x -= u_wind * u_time * 0.35;
	vec2 w = vec2(fbm(q * 1.9 + 13.7), fbm(q * 1.9 - 71.3)) - 0.5;
	float n = fbm(q + w * u_turb * (2.4 + 2.2 * min(v, 1.4) + 1.6 * u_burst));

	// The field: heat falls off up the column, and the noise is SUBTRACTED with a
	// weight that grows with height - near the bed it only textures, near the tip
	// it cuts clean through, pinching tongues loose. n*n keeps the cuts sharp.
	float flame = env * pow(max(1.0 - v, 0.0), 0.85) * 1.30
		- (n * n) * (0.22 + 1.05 * min(v, 1.5));
	// Soft ceiling just inside the quad top: an extreme surge dissolves instead of
	// clipping flat on the quad edge.
	float top = u_base + u_half.y;
	flame *= 1.0 - smoothstep(top - 0.25, top - 0.04, yu);

	float body = smoothstep(0.0, 0.10, flame); // soft outer edge
	// Inner temperature, CHURNED by the same turbulence that cuts the edge - without
	// this a roaring column saturates into one flat white shape; with it the interior
	// streams and boils, white only in the hottest threads.
	float heat = clamp(flame * 1.7, 0.0, 1.0) * (0.55 + 0.55 * n);

	// Slow whole-flame breathing + fine shimmer, both spatial so the field never
	// blinks in unison.
	float flick = 0.82 + 0.12 * sin(u_time * 8.2 + p.x * 6.0 + n * 7.0)
		+ 0.06 * sin(u_time * 2.1 + p.x * 1.7);

	// Blackbody-ish ramp: dim deep red edge -> orange -> yellow -> white-hot core.
	vec3 col = mix(vec3(0.42, 0.02, 0.0), vec3(1.0, 0.32, 0.02), smoothstep(0.0, 0.45, heat));
	col = mix(col, vec3(1.0, 0.74, 0.22), smoothstep(0.45, 0.84, heat));
	col = mix(col, vec3(1.0, 0.97, 0.86), smoothstep(0.84, 1.0, heat));

	float a = body * (0.30 + 0.70 * heat) * flick;

	// The ember bed: a thin glowing seam along the base, bright under hot sources,
	// spilling a little light below the bed line too.
	float seam = exp(-abs(yu) * (yu < 0.0 ? 22.0 : 9.0));
	vec3 bed = vec3(1.0, 0.38, 0.06) * seam * (0.10 + 0.30 * env) * flick;

	COLOR = vec4((col * a + bed) * u_gain * (1.0 + 0.35 * u_burst), a) * tint;
}
