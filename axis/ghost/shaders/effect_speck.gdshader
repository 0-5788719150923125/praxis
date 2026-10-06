// A dark speck of an effect (see scripts/effects.gd): a fly, a gnat - laid over what is behind it,
// never added to it, so it shows dark on the cloth and against the light. Its body fills the middle
// third of its quad, which the vertex smears along its motion. `glow` is how dark it is.
shader_type spatial;
render_mode unshaded, blend_mix, depth_draw_never, cull_disabled, shadows_disabled, skip_vertex_transform;

#include "res://shaders/effect_sprite.gdshaderinc"

void fragment() {
	vec2 p = corner * 2.0 - 1.0;
	float body = 1.0 - smoothstep(0.24, 0.34, length(vec2(p.x * 1.1, p.y)));
	ALBEDO = tint.rgb;
	ALPHA = clamp(body * fade * glow, 0.0, 1.0);
}
