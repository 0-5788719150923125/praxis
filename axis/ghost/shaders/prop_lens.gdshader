// A modeled thing's SOLID glass or clear crystal (see scripts/props.gd) - a crystal ball, a lens, a
// clear point: what is behind it is seen through it, bent and turned over as a lens turns it, and
// its skin catches the flames. Hollow glass (a jar, a bottle) is prop_glass, which a flame inside
// can be seen through.
shader_type spatial;
render_mode blend_mix, depth_draw_always, cull_back, diffuse_burley, specular_schlick_ggx;

uniform sampler2D screen : hint_screen_texture, filter_linear_mipmap;
uniform vec3 color : source_color = vec3(0.9, 0.93, 0.95);
uniform vec3 color2 : source_color = vec3(0.3);
uniform float polish : hint_range(0.0, 1.0) = 0.9;
uniform float wear : hint_range(0.0, 1.0) = 0.0;
uniform float pattern : hint_range(0.0, 1.0) = 0.0;
uniform float clarity : hint_range(0.0, 1.0) = 0.9;
#include "res://shaders/prop_common.gdshaderinc"

void fragment() {
	vec2 o = ornament(UV, UV2);
	float facing = clamp(dot(NORMAL, VIEW), 0.0, 1.0);
	float fr = pow(1.0 - facing, 3.0);
	float frost = clamp(max(1.0 - clarity, o.y * orn_depth), 0.0, 1.0);
	// TURNED OVER: the line of sight is bent toward the middle and past it, so the edge of the
	// thing shows what lies beyond its far side - the further away it stands, the smaller the bend
	// is on the screen
	float k = (0.02 + 0.025 * facing) / max(-VERTEX.z, 0.15);
	vec3 seen = textureLod(screen, clamp(SCREEN_UV + NORMAL.xy * vec2(-1.0, 1.0) * k, vec2(0.0), vec2(1.0)), frost * 4.0).rgb;
	vec3 tint = mix(vec3(1.0), color, 0.55);
	// milky where it is not clear: some of its own color, lit
	ALBEDO = color * 0.6 * frost;
	EMISSION = seen * tint * (1.0 - fr * 0.55) * (1.0 - frost * 0.7);
	ROUGHNESS = mix(0.02 + (1.0 - polish) * 0.15, 0.45, frost);
	METALLIC = 0.0;
	SPECULAR = 0.6;
	if (play == 5) {
		// RAINBOW: a thin film on its skin (aura quartz) - the colors in what it reflects, and less
		// let through of what is behind it
		ALBEDO = mix(ALBEDO, film(NORMAL, VIEW, color), 0.85);
		METALLIC = 0.6;
		EMISSION *= 0.6;
	}
	if (orn_paint > 0.0) {
		ALBEDO = mix(ALBEDO, orn_color, o.y * orn_paint);
		EMISSION *= 1.0 - o.y * orn_paint;
	}
	NORMAL = bump(NORMAL, VERTEX, o.x * orn_depth * 0.0004);
	ALPHA = 1.0;
}
