// THE INTRO, THROWN FAR OUT OF FOCUS (the user, 2026-10-06: "an extremely strong blur, such that
// nothing is really visible in the scene except for the colors"): the whole frame read back through
// a wide Gaussian while the show's name is up over it. The lens's own depth of field never got there -
// at its softest every thing on the table could still be named. Each pixel is read from the screen's
// mips, which Godot builds level by level with a Gaussian, at the level whose blur matches - through
// a B-spline, so the coarse level's texels never show as blocks - and between two levels, so the blur
// can ease away smoothly as the focus is pulled.
shader_type canvas_item;
render_mode unshaded;

uniform sampler2D screen : hint_screen_texture, filter_linear_mipmap, repeat_disable;
// the blur's standard deviation, as a share of the frame's height; 0 leaves the frame as it is
uniform float sigma = 0.0;
// the blur one mip level carries, read through the B-spline, in that level's texels - measured off
// an edge (tests/intro_blur_check.gd): at 2.07 the blur drawn is the blur asked for, at 720 and 1080
uniform float level_sigma = 2.07;

vec3 bspline(vec2 uv, int level) {
	vec2 size = vec2(textureSize(screen, level));
	vec2 p = uv * size - 0.5;
	vec2 i = floor(p);
	vec2 f = p - i;
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
	vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
	vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
	vec2 w3 = f3 / 6.0;
	vec2 g0 = w0 + w1;
	vec2 g1 = w2 + w3;
	// two texels a fetch: the bilinear filter weighs each pair as the spline does
	vec2 t0 = (i - 0.5 + w1 / g0) / size;
	vec2 t1 = (i + 1.5 + w3 / g1) / size;
	float l = float(level);
	return (textureLod(screen, t0, l).rgb * g0.x + textureLod(screen, vec2(t1.x, t0.y), l).rgb * g1.x) * g0.y
		+ (textureLod(screen, vec2(t0.x, t1.y), l).rgb * g0.x + textureLod(screen, t1, l).rgb * g1.x) * g1.y;
}

void fragment() {
	// the blur wanted, in pixels, and the mip level that carries it
	float s = sigma / SCREEN_PIXEL_SIZE.y;
	float lod = clamp(log2(max(s / level_sigma, 1.0)), 0.0, 12.0);
	int l0 = int(floor(lod));
	vec3 c = bspline(SCREEN_UV, l0);
	if (lod > float(l0)) {
		c = mix(c, bspline(SCREEN_UV, l0 + 1), lod - float(l0));
	}
	COLOR = vec4(c, 1.0);
}
