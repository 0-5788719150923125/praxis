// A modeled thing's GLASS (see scripts/props.gd): tinted, see-through at its face and silvered at
// its edges, frosted as its clarity falls - and an ornament on glass is etched, so it frosts too.
shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

uniform vec3 color : source_color = vec3(0.85, 0.9, 0.88);
uniform vec3 color2 : source_color = vec3(0.3);
uniform float polish : hint_range(0.0, 1.0) = 0.9;
uniform float wear : hint_range(0.0, 1.0) = 0.0;
uniform float pattern : hint_range(0.0, 1.0) = 0.0;
uniform float clarity : hint_range(0.0, 1.0) = 0.85;
#include "res://shaders/prop_common.gdshaderinc"

void fragment() {
	vec2 o = ornament(UV, UV2);
	float fr = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 3.0);
	float frosted = clamp(max(1.0 - clarity, o.y * orn_depth), 0.0, 1.0);
	vec3 albedo = mix(color * 0.3, mix(color, vec3(0.92), 0.45), frosted);
	float alpha = 0.12 + 0.55 * fr + frosted * 0.45;
	float metal = 0.0;
	if (play == 5) {
		// RAINBOW: an iridescent film on it, as on old buried glass
		albedo = mix(albedo, film(NORMAL, VIEW, color), 0.8);
		alpha = max(alpha, 0.35);
		metal = 0.5;
	}
	if (orn_paint > 0.0) {
		albedo = mix(albedo, orn_color, o.y * orn_paint);
		alpha = max(alpha, o.y * orn_paint * 0.9);
	}
	ALBEDO = albedo;
	ALPHA = clamp(alpha, 0.0, 0.95);
	METALLIC = metal;
	ROUGHNESS = mix(0.03 + (1.0 - polish) * 0.2, 0.5, frosted);
	SPECULAR = 0.55;
	NORMAL = bump(NORMAL, VERTEX, o.x * orn_depth * 0.0004);
}
