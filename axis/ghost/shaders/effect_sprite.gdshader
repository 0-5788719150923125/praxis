// A glowing sprite of an effect (see scripts/effects.gd): a mote or a burst's particle, added to the
// light already there - brighter than white, so the stage's bloom spreads it.
shader_type spatial;
render_mode unshaded, blend_add, depth_draw_never, cull_disabled, shadows_disabled, skip_vertex_transform;

#include "res://shaders/effect_sprite.gdshaderinc"

void fragment() {
	vec2 p = corner * 2.0 - 1.0;
	float r = length(p);
	float a;
	vec3 col = tint.rgb;
	if (shape == 1) {
		// a streak: hot along its middle line, fading at its ends
		a = exp(-p.x * p.x * 12.0) * (1.0 - smoothstep(0.55, 1.0, abs(p.y)));
		col = mix(col, vec3(1.0, 0.97, 0.9), exp(-p.x * p.x * 40.0) * 0.3);
	} else if (shape == 2) {
		// a flame: a tongue, round below and drawn out above, whiter at its root
		vec2 q = vec2(p.x * 1.7, p.y + 0.35);
		float body = 1.0 - smoothstep(0.0, 0.85, length(vec2(q.x, q.y > 0.0 ? q.y * 0.55 : q.y * 1.5)));
		a = body * body;
		col = mix(col, vec3(1.0, 0.95, 0.82), smoothstep(0.55, 1.0, body) * 0.8);
	} else if (shape == 4) {
		// a star: four thin rays round a bright core
		a = exp(-abs(p.x) * 26.0) * exp(-abs(p.y) * 2.4) + exp(-abs(p.y) * 26.0) * exp(-abs(p.x) * 2.4)
			+ exp(-r * r * 24.0);
	} else if (shape == 5) {
		// a flake: a soft disc, a little six-lobed
		a = (1.0 - smoothstep(0.35, 0.95, r)) * (0.82 + 0.18 * cos(atan(p.y, p.x) * 6.0));
	} else {
		// an orb: a hot core in a soft halo
		a = exp(-r * r * 10.0) + 0.3 * exp(-r * r * 2.2);
		a *= 1.0 - smoothstep(0.8, 1.0, r);
		col = mix(col, vec3(1.0), exp(-r * r * 40.0) * 0.35);
	}
	ALBEDO = col * glow;
	ALPHA = clamp(a * fade, 0.0, 1.0);
}
