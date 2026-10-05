// What every modeled thing's surface shares (see scripts/props.gd): noise in the thing's own
// space, relief without tangents, the ORNAMENT - a motif laid out round a part, worked in as
// relief, paint or inlay - and what the PLAY OF LIGHT needs. Included by prop.gdshader,
// prop_glass.gdshader and prop_lens.gdshader.

// Props.ORNAMENTS: 0 none, 1 bands, 2 flutes, 3 dots, 4 stars, 5 moons, 6 zigzag, 7 scales,
// 8 spiral, 9 lattice, 10 hammered
uniform int orn = 0;
uniform float orn_count = 8.0;
uniform float orn_depth = 0.5;
uniform vec3 orn_color : source_color = vec3(0.8, 0.62, 0.25);
uniform float orn_paint = 0.0;
uniform vec2 orn_zone = vec2(0.0, 1.0);
uniform vec2 dims = vec2(20.0, 10.0);      // the part's girth and height, centimeters
uniform float grain = 1.0;                 // the material's pattern size, centimeters
uniform float seed = 0.0;
// Props.PLAYS: 0 none, 1 silk, 2 flash, 3 fire, 4 glitter, 5 rainbow, 6 glow
uniform int play = 0;
// a crystal's banding or a stone's veins curve round in rings (agate, malachite)
uniform float rings = 0.0;

varying vec3 opos;          // the thing's own space, in pattern units
varying vec3 cpos;          // ...in centimeters
varying vec3 lpos;          // ...in centimeters from the middle of its own copy (Props.Tris.placed)
varying flat float tag;     // the copy's own number, 0..1

void vertex() {
	cpos = VERTEX * 100.0;
	opos = cpos / max(grain, 0.05) + vec3(seed * 17.31, seed * 3.7, seed * 9.13);
	lpos = (VERTEX - CUSTOM0.xyz) * 100.0;
	tag = CUSTOM0.w;
}

float hash3(vec3 p) {
	p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vnoise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(hash3(i), hash3(i + vec3(1.0, 0.0, 0.0)), f.x),
			mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
		mix(mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), f.x),
			mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}

float fbm(vec3 p) {
	float s = 0.0;
	float a = 0.5;
	for (int i = 0; i < 4; i++) {
		s += a * vnoise(p);
		p = p * 2.07 + vec3(1.7, 9.2, 3.4);
		a *= 0.5;
	}
	return s / 0.9375;
}

// the nearest and second-nearest cell centers' distances, and a number of the nearest cell's own
vec3 cellsid(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	float d1 = 8.0;
	float d2 = 8.0;
	float id = 0.0;
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			for (int z = -1; z <= 1; z++) {
				vec3 g = vec3(float(x), float(y), float(z));
				vec3 o = vec3(hash3(i + g), hash3(i + g + 17.1), hash3(i + g + 33.7));
				float d = length(g + o - f);
				if (d < d1) {
					d2 = d1;
					d1 = d;
					id = hash3(i + g + 51.3);
				} else if (d < d2) {
					d2 = d;
				}
			}
		}
	}
	return vec3(d1, d2, id);
}

vec2 cells(vec3 p) {
	return cellsid(p).xy;
}

vec3 rgb2hsv(vec3 c) {
	vec4 k = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, k.wz), vec4(c.gb, k.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}

vec3 hsv2rgb(vec3 c) {
	vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
	return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// a direction of each copy's own, in the thing's space
vec3 own_dir(float salt) {
	return normalize(vec3(hash3(vec3(tag * 31.7 + salt)), hash3(vec3(tag * 57.3 + salt)), hash3(vec3(tag * 83.9 + salt))) - 0.5 + vec3(1e-4));
}

// a silky stone's fibers, lying mostly flat
vec3 fibers() {
	vec3 f = own_dir(0.0);
	return normalize(vec3(f.x, f.y * 0.3, f.z) + vec3(1e-4));
}

// THE BANDING's phase: up the part in straight bands, or round in rings about the middle of its own
// copy (a little off it), warped by [param warp] - or, in a silky stone, along its fibers, so its
// bands lie across them
float banding(float warp) {
	if (play == 1) {
		return dot(lpos, fibers()) / max(grain, 0.05) * 2.6 + warp * 0.8;
	}
	if (rings > 0.5) {
		// the rings wander as the hollow they grew in did: the space they lie in is bent by a few
		// centimeters, and they ripple a little - a bend as fine as the bands blotched them
		vec3 off = own_dir(5.0) * grain * 1.5;
		vec3 w = vec3(fbm(lpos * 0.22 + vec3(tag * 7.0)), fbm(lpos * 0.22 + vec3(5.2, tag * 7.0, 1.3)),
			fbm(lpos * 0.22 + vec3(9.1, 2.8, tag * 7.0))) - 0.5;
		return length((lpos + w * 4.0 - off) * vec3(1.0, 1.35, 1.0)) / max(grain, 0.05) * 2.6 + fbm(lpos * 1.1 + vec3(3.3)) * 1.2;
	}
	return opos.y * 2.6 + warp * 5.0;
}

// A THIN FILM's color - an oily rainbow, its hue set by how thick the film is there and the angle it
// is seen at. Over a colored body the hues keep near its own (paua's blues and greens, not a
// tie-dye of every color); over black, white or gray they run the whole rainbow; paler over a pale
// body
vec3 film(vec3 n, vec3 v, vec3 body) {
	vec3 own = rgb2hsv(body);
	float thick = fbm(opos * 0.8 + vec3(5.0)) * 1.7 + (1.0 - clamp(dot(n, v), 0.0, 1.0)) * 0.75;
	float spread = mix(1.0, 0.4, smoothstep(0.15, 0.4, own.y));
	float h = fract(own.x + (fract(thick) - 0.5) * spread);
	float sat = mix(0.85, 0.35, clamp(dot(body, vec3(0.299, 0.587, 0.114)), 0.0, 1.0));
	return hsv2rgb(vec3(h, sat, 1.0));
}

// a small mirror's facing, its own for each [param id]: a nudge to the surface's normal, in the
// thing's space, up to [param most] long
vec3 tipped(float id, float most) {
	return (vec3(hash3(vec3(id * 31.0)), hash3(vec3(id * 57.0)), hash3(vec3(id * 91.0))) - 0.5) * 2.0 * most;
}

// a five-pointed star's distance (negative inside)
float star5(vec2 p, float r, float rf) {
	const vec2 k1 = vec2(0.809016994375, -0.587785252292);
	const vec2 k2 = vec2(-0.809016994375, -0.587785252292);
	p.x = abs(p.x);
	p -= 2.0 * max(dot(k1, p), 0.0) * k1;
	p -= 2.0 * max(dot(k2, p), 0.0) * k2;
	p.x = abs(p.x);
	p.y -= r;
	vec2 ba = rf * vec2(-k1.y, k1.x) - vec2(0.0, 1.0);
	float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
	return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

// RELIEF FROM A HEIGHT, in meters, without tangents: the surface's own screen derivatives. Called
// on every pixel, never inside a branch some pixels skip - a derivative there is undefined - and
// the surface kept as it is where it is seen edge-on: a pore there normalized a zero vector, and
// the NaN it made reached the reflection probe and blacked out every thing that reflects it
vec3 bump(vec3 n, vec3 pos, float h) {
	vec3 dpdx = dFdx(pos);
	vec3 dpdy = dFdy(pos);
	float dhx = dFdx(h);
	float dhy = dFdy(h);
	vec3 r1 = cross(dpdy, n);
	vec3 r2 = cross(n, dpdx);
	float det = dot(dpdx, r1);
	vec3 g = sign(det) * (dhx * r1 + dhy * r2);
	vec3 b = abs(det) * n - g;
	return dot(b, b) > 1e-30 ? normalize(b) : n;
}

// THE ORNAMENT: x its relief 0..1, y where it is (paint and inlay go there). A motif is laid out
// in cells as wide around the part as they are tall, so a star is a star on a tall vase and on a
// squat bowl alike.
vec2 ornament(vec2 uv, vec2 uv2) {
	if (orn == 0) {
		return vec2(0.0);
	}
	if (orn == 10) {
		vec2 c = cells(cpos * (0.9 + orn_count * 0.08) + vec3(seed));
		float dimple = 1.0 - c.x * c.x;
		return vec2(dimple, 0.0);
	}
	float z0 = min(orn_zone.x, orn_zone.y);
	float z1 = max(orn_zone.x, orn_zone.y);
	float y = uv2.x;
	float inzone = smoothstep(z0 - 0.004, z0 + 0.004, y) * (1.0 - smoothstep(z1 - 0.004, z1 + 0.004, y));
	if (inzone <= 0.0) {
		return vec2(0.0);
	}
	float v = (y - z0) / max(z1 - z0, 0.001);
	float u = uv.x;
	float n = max(orn_count, 1.0);
	float zh = max(dims.y * (z1 - z0), 0.1);
	float cw = max(dims.x / n, 0.05);
	float rows = max(1.0, floor(zh / cw + 0.5));
	vec2 cell = vec2(u * n, v * rows);
	vec2 p = vec2(fract(cell.x) - 0.5, (fract(cell.y) - 0.5) * (zh / rows) / cw);
	float h = 0.0;
	if (orn == 1) {
		float t = fract(v * n);
		h = 1.0 - smoothstep(0.06, 0.14, abs(t - 0.5));
	} else if (orn == 2) {
		h = 0.5 + 0.5 * cos(6.2831853 * u * n);
	} else if (orn == 3) {
		h = 1.0 - smoothstep(0.16, 0.21, length(p));
	} else if (orn == 4) {
		h = 1.0 - smoothstep(-0.01, 0.02, star5(vec2(p.x, -p.y), 0.32, 0.42));
	} else if (orn == 5) {
		float d = max(length(p) - 0.28, -(length(p - vec2(0.11, 0.07)) - 0.24));
		h = 1.0 - smoothstep(-0.01, 0.02, d);
	} else if (orn == 6) {
		float line = 0.5 + 0.36 * (abs(fract(u * n) - 0.5) * 4.0 - 1.0);
		float w = clamp(0.35 / zh, 0.03, 0.2);
		h = 1.0 - smoothstep(w, w * 1.8, abs(v - line));
	} else if (orn == 7) {
		vec2 q = vec2(fract(cell.x + 0.5 * mod(floor(cell.y), 2.0)) - 0.5, p.y);
		float d = length(q - vec2(0.0, 0.42 * (zh / rows) / cw));
		h = smoothstep(0.38, 0.47, d) * (1.0 - smoothstep(0.5, 0.56, d)) + (1.0 - smoothstep(0.0, 0.5, d)) * 0.25;
	} else if (orn == 8) {
		h = 1.0 - smoothstep(0.08, 0.16, abs(fract(cell.x + cell.y) - 0.5));
	} else if (orn == 9) {
		float a = abs(fract(cell.x + cell.y) - 0.5);
		float b = abs(fract(cell.x - cell.y) - 0.5);
		h = 1.0 - smoothstep(0.05, 0.11, min(a, b));
	}
	return vec2(h, h) * inzone;
}

