// Fog of an effect (see scripts/effects.gd): one FogVolume's density, rolling through its box as a
// function of SHOW TIME - the noise is carried by a slow wind and turns over itself - thickest at its
// floor and thinning toward its height, soft at its sides. The scene's lights light it (Godot's own
// volumetric fog), so candles and lit motes glow inside it.
shader_type fog;

uniform float show_time = 0.0;
uniform vec3 color : source_color = vec3(0.75, 0.78, 0.82);
uniform float density = 1.0;
uniform float rolls = 0.5;
uniform vec3 wind = vec3(0.03, 0.01, 0.0);   // m/s the fog is carried
uniform float grain = 0.25;                  // meters: the size of its swirls
uniform float floor_y = 0.0;                 // its floor, world meters
uniform float height = 0.25;                 // how high it reaches above its floor, meters
uniform float edge = 0.15;                   // how far in from its sides it thickens, meters
uniform float glow = 0.0;                    // light of its own

float fog_hash(vec3 p) {
	p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float fog_noise(vec3 x) {
	vec3 i = floor(x);
	vec3 f = fract(x);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(fog_hash(i), fog_hash(i + vec3(1.0, 0.0, 0.0)), f.x),
			mix(fog_hash(i + vec3(0.0, 1.0, 0.0)), fog_hash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
		mix(mix(fog_hash(i + vec3(0.0, 0.0, 1.0)), fog_hash(i + vec3(1.0, 0.0, 1.0)), f.x),
			mix(fog_hash(i + vec3(0.0, 1.0, 1.0)), fog_hash(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}

float fog_fbm(vec3 p) {
	float a = 0.5;
	float s = 0.0;
	for (int i = 0; i < 4; i++) {
		s += a * fog_noise(p);
		p = p * 2.03 + vec3(1.7, 9.2, 3.1);
		a *= 0.5;
	}
	return s;
}

void fog() {
	vec3 w = WORLD_POSITION;
	vec3 q = (w - wind * show_time) / max(grain, 0.02);
	// ROLLING: the noise's own space turns slowly over itself, so the fog billows as it drifts
	float t = show_time;
	q += rolls * 1.2 * vec3(sin(q.y * 0.71 + t * 0.13), sin(q.z * 0.63 + t * 0.11) * 0.6, sin(q.x * 0.52 + t * 0.17));
	// billows (the noise at its grain) carved by finer wisps, so the fog has shape and not only depth
	float n = fog_fbm(q + vec3(0.0, -t * 0.02 / max(grain, 0.02), 0.0));
	n = n * 0.75 + 0.25 * fog_noise(q * 3.1 + vec3(t * 0.05, 0.0, 0.0));
	float h = clamp((w.y - floor_y) / max(height, 0.01), 0.0, 2.0);
	float layer = 1.0 - smoothstep(0.25, 1.0, h);
	// SOFT AT ITS SIDES, never at its floor or its top: a layer lying on the cloth thickens toward the
	// cloth (the distance to the nearest face would thin it to nothing exactly there)
	vec2 e = vec2(edge) / max(SIZE.xz, vec2(0.01));
	float side = smoothstep(0.0, e.x, UVW.x) * smoothstep(0.0, e.x, 1.0 - UVW.x)
		* smoothstep(0.0, e.y, UVW.z) * smoothstep(0.0, e.y, 1.0 - UVW.z);
	// BILLOWS with gaps between them: thick where the noise is high, clear where it is low
	float d = density * layer * side * smoothstep(0.44, 0.72, n + 0.15 * (1.0 - h));
	DENSITY = d;
	ALBEDO = color;
	EMISSION = color * glow * d;
}
