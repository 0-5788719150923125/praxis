// What every sprite of an effect shares (see scripts/effects.gd): a quad turned to the camera,
// sized in meters, at a place that is a function of SHOW TIME. A mote is POSED (the CPU put it where
// it is now: its place, its size in the custom data's w, its brightness in the color's alpha). A
// burst's particle FLIES: born at a time and place with a velocity (the transform's first column -
// nothing else is in it), pulled by gravity and slowed by the air in closed form, so any time can be
// asked for and none is stepped to. Included by effect_sprite.gdshader and effect_smoke.gdshader,
// both with render_mode skip_vertex_transform: VERTEX is set in view space here.

uniform float show_time = 0.0;
uniform bool posed = false;
uniform vec3 gravity = vec3(0.0, -2.0, 0.0);   // m/s2
uniform float drag = 1.0;                      // 1/s
uniform float glow = 2.0;                      // brighter than white blooms
uniform int shape = 0;                         // 0 orb, 1 streak, 2 flame, 3 puff, 4 star, 5 flake
uniform float stretch = 0.045;                 // a streak's length per m/s, seconds
uniform float grow = 0.0;                      // sizes added over a life
uniform float waver = 0.0;                     // meters a second of sideways wandering
uniform float twinkle = 0.0;                   // how much its brightness flutters

varying vec4 tint;
varying vec2 corner;
varying float fade;

void vertex() {
	vec3 pos;
	vec3 vel = vec3(0.0);
	float size;
	float hide = 0.0;
	float turn = 0.0;
	fade = 1.0;
	if (posed) {
		pos = MODEL_MATRIX[3].xyz;
		size = INSTANCE_CUSTOM.w;
		fade = COLOR.a;
		hide = COLOR.a <= 0.002 ? 1.0 : 0.0;
	} else {
		vec3 p0 = MODEL_MATRIX[3].xyz;
		vec3 v0 = MODEL_MATRIX[0].xyz;
		float born = INSTANCE_CUSTOM.x;
		float life = max(INSTANCE_CUSTOM.y, 0.01);
		float seed = INSTANCE_CUSTOM.z;
		size = INSTANCE_CUSTOM.w;
		float age = show_time - born;
		float u = age / life;
		hide = (age < 0.0 || u > 1.0) ? 1.0 : 0.0;
		age = max(age, 0.0);
		// linear drag, in closed form: where it is and how fast it goes at this age
		float k = max(drag, 0.001);
		float e = exp(-k * age);
		pos = p0 + v0 * (1.0 - e) / k + gravity * (age - (1.0 - e) / k) / k;
		vel = v0 * e + gravity * (1.0 - e) / k;
		pos += waver * age * vec3(sin(age * 6.3 + seed * 40.0), 0.4 * sin(age * 4.1 + seed * 13.0), cos(age * 5.7 + seed * 23.0));
		size *= 1.0 + grow * u;
		fade = smoothstep(0.0, 0.06, u) * (1.0 - smoothstep(0.5, 1.0, u));
		fade *= 1.0 - twinkle * 0.5 * (1.0 + sin(age * 29.0 + seed * 91.0));
		turn = seed * 6.2831 + age * (seed - 0.5) * 6.0;
	}
	vec3 view = (VIEW_MATRIX * vec4(pos, 1.0)).xyz;
	vec2 c = VERTEX.xy;
	if (shape == 1) {
		// A STREAK lies along its motion as the camera sees it, longer the faster it goes
		vec2 d = (VIEW_MATRIX * vec4(vel, 0.0)).xy;
		float len = length(d);
		vec2 dir = len > 1e-5 ? d / len : vec2(0.0, 1.0);
		vec2 side = vec2(-dir.y, dir.x);
		view.xy += dir * c.y * (size + len * stretch) + side * c.x * size;
	} else if (shape == 2) {
		// A FLAME stands up from its base
		view.xy += vec2(c.x * size, (c.y + 0.4) * size * 2.0);
	} else {
		float cs = cos(turn);
		float sn = sin(turn);
		view.xy += mat2(vec2(cs, sn), vec2(-sn, cs)) * (c * size);
	}
	// a sprite not alive now is put behind the camera, where nothing is drawn
	VERTEX = hide > 0.5 ? vec3(0.0, 0.0, 1.0) : view;
	NORMAL = vec3(0.0, 0.0, 1.0);
	tint = vec4(COLOR.rgb, 1.0);
	corner = UV;
}
