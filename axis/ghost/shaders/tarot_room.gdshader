shader_type spatial;
render_mode unshaded;

// THE ROOM PAST THE TABLE, out of focus as a lens focused on the table sees it: every point is a
// disc of the picture - a real lens's bokeh, so a bright reflection opens into a soft round spot -
// read at the picture's full resolution. Shrinking the picture to blur it (to an eighth and back)
// left the dozen rows of it the camera sees past the table magnified into blocks (feedback 0006:
// "very pixelated, and not because of blurring").

uniform sampler2D picture : source_color, filter_linear_mipmap, repeat_disable;
// the room's color where there is no picture yet
uniform vec3 tint : source_color = vec3(1.0);
uniform float has_picture = 0.0;
// the disc's radius in the picture's own UV, per axis, so it is round on the quad
uniform vec2 radius = vec2(0.004, 0.006);

const int TAPS = 48;
const float GOLDEN = 2.39996323;

void fragment() {
	vec3 c = tint;
	if (has_picture > 0.5) {
		vec2 size = vec2(textureSize(picture, 0));
		// each tap stands for a patch of the disc this many texels across: read it from the mip
		// that averages that much, or from whatever the screen already shrinks it to
		float patch = max(radius.x * size.x, radius.y * size.y) * sqrt(3.14159265 / float(TAPS));
		float shrink = max(length(dFdx(UV * size)), length(dFdy(UV * size)));
		float lod = max(0.0, log2(max(patch, shrink)));
		vec3 sum = vec3(0.0);
		for (int i = 0; i < TAPS; i++) {
			float r = sqrt((float(i) + 0.5) / float(TAPS));
			float a = float(i) * GOLDEN;
			sum += textureLod(picture, clamp(UV + vec2(cos(a), sin(a)) * r * radius, vec2(0.0), vec2(1.0)), lod).rgb;
		}
		c = sum / float(TAPS);
	}
	ALBEDO = c;
}
