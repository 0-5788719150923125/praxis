// A modeled thing's surface (see scripts/props.gd): one shader for every opaque material, its
// look procedural - noise in the thing's own space, so nothing tiles and nothing needs a texture.
// Glass is its own shader (prop_glass); what both share is prop_common.gdshaderinc.
shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

// Props.MATERIALS: 0 painted, 1 metal, 2 wax, 4 crystal, 5 ceramic, 6 stone, 7 wood, 8 paper,
// 9 leather, 10 fabric, 11 bone, 12 plant
uniform int kind = 0;
uniform vec3 color : source_color = vec3(0.7);
uniform vec3 color2 : source_color = vec3(0.3);
uniform float polish : hint_range(0.0, 1.0) = 0.5;
uniform float wear : hint_range(0.0, 1.0) = 0.0;
uniform float pattern : hint_range(0.0, 1.0) = 0.5;
uniform float clarity : hint_range(0.0, 1.0) = 0.5;
// a lit candle's wax: its flame's brightness now (0 unlit), which glows in the top of it
uniform float flame = 0.0;
#include "res://shaders/prop_common.gdshaderinc"

void fragment() {
	float n1 = fbm(opos * 0.6);
	float n2 = fbm(opos * 3.1 + vec3(7.0));
	vec2 o = ornament(UV, UV2);
	vec3 albedo = color;
	float rough = 1.0 - polish * 0.92;
	float metal = 0.0;
	float spec = 0.5;
	float relief = o.x * orn_depth * (orn == 10 ? 0.0005 : 0.0008);
	float sss = 0.0;
	vec3 back = vec3(0.0);
	float coat = 0.0;
	float coat_rough = 0.2;
	float rim = 0.0;
	vec3 emit = vec3(0.0);
	if (kind == 1) {
		// METAL: tarnish and patina gather where the surface is worked and where noise says
		metal = 1.0;
		albedo = color * (0.9 + 0.2 * n1);
		rough = clamp(rough + (n2 - 0.5) * 0.14, 0.04, 1.0);
		float groove = o.x * (1.0 - orn_paint);
		float pm = smoothstep(0.5, 0.85, n1 * 0.55 + n2 * 0.45 + wear * 0.45 - 0.2 + groove * 0.5) * wear;
		albedo = mix(albedo, color2, pm);
		metal = mix(1.0, 0.15, pm);
		rough = mix(rough, 0.8, pm);
		if (pattern > 0.0) {
			vec2 c = cells(opos * 1.3);
			relief += (1.0 - c.x * c.x) * pattern * 0.00035;
		}
	} else if (kind == 2) {
		// WAX: light goes into it and comes back out warm
		albedo = color * (0.93 + 0.1 * n1);
		rough = 0.62 - polish * 0.35;
		sss = 0.25 + 0.3 * clarity;
		back = color * (0.04 + 0.1 * clarity);
		spec = 0.35;
		// THE TOP OF A BURNING CANDLE GLOWS: its flame's light caught in the wax under it (the flame's
		// own light leaves its candle alone, or a low flame blew out what held it)
		emit = color * flame * (0.08 + 0.3 * clarity) * smoothstep(0.55, 1.0, UV2.x);
	} else if (kind == 4) {
		// CRYSTAL: a milky inside, banding, and a glassy skin that catches the flames
		float cloud = fbm(opos * 0.8 + vec3(3.0));
		albedo = mix(color, mix(color, vec3(1.0), 0.5), cloud * (1.0 - clarity));
		float band = smoothstep(0.35, 0.65, sin(banding(n1)) * 0.5 + 0.5) * pattern;
		albedo = mix(albedo, color2, band * 0.75);
		float fr = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 3.0);
		albedo *= mix(0.55 + 0.25 * clarity, 1.15, fr);
		rough = 0.03 + (1.0 - polish) * 0.3;
		sss = 0.5 + 0.3 * clarity;
		back = albedo * (0.4 + 0.5 * clarity);
		spec = 0.9;
		coat = 0.7;
		coat_rough = 0.03;
	} else if (kind == 5) {
		// CERAMIC: glaze (polish), speckle (pattern), crazing (wear)
		albedo = color * (0.94 + 0.1 * n1);
		albedo = mix(albedo, color2, smoothstep(0.55, 0.85, n1) * 0.35 * (1.0 - pattern * 0.5));
		float speck = step(0.8, vnoise(opos * 9.0)) * pattern;
		albedo *= 1.0 - speck * 0.55;
		vec2 c = cells(opos * 1.6);
		float craze = (1.0 - smoothstep(0.0, 0.05, c.y - c.x)) * wear;
		albedo *= 1.0 - craze * 0.45;
		relief -= craze * 0.0001;
		rough = mix(0.8, 0.07, polish);
		coat = polish * 0.8;
		coat_rough = 0.08;
	} else if (kind == 6) {
		// STONE: veins (pattern) - running through it, or round in rings - and flecks (wear)
		float t = (rings > 0.5 || play == 1) ? banding(fbm(opos * 0.7)) * 0.5 : opos.x * 0.8 + opos.y * 0.3 + fbm(opos * 0.7) * 4.0;
		float vein = 1.0 - smoothstep(0.0, 0.06 + 0.12 * (1.0 - pattern), abs(sin(t)));
		albedo = mix(color * (0.88 + 0.24 * n2), color2, vein * pattern);
		float fleck = step(0.86, vnoise(opos * 6.0)) * wear;
		albedo = mix(albedo, color2 * 0.6, fleck);
		rough = mix(0.85, 0.12, polish);
		coat = polish * 0.4;
	} else if (kind == 7) {
		// WOOD: rings about the part's axis, and the fibers along it
		float d = length(opos.xz) * 0.9 + fbm(opos * vec3(0.35, 0.08, 0.35)) * 2.2 + opos.y * 0.04;
		float ring = smoothstep(0.15, 0.85, abs(fract(d) * 2.0 - 1.0));
		float fiber = fbm(opos * vec3(2.5, 0.25, 2.5));
		albedo = mix(color, color2, ring * pattern * 0.8) * (0.93 + 0.14 * fiber);
		rough = mix(0.82, 0.22, polish);
		coat = polish * 0.5;
	} else if (kind == 8) {
		// PAPER: stained where it has been handled
		albedo = color * (0.93 + 0.1 * n2);
		albedo = mix(albedo, color2, smoothstep(0.55, 0.8, fbm(opos * 0.5 + vec3(3.0))) * wear);
		rough = 0.9;
		spec = 0.3;
	} else if (kind == 9) {
		// LEATHER: a pebbled grain, worn paler
		vec2 c = cells(opos * 3.0);
		relief += c.x * 0.00025;
		albedo = color * (0.85 + 0.25 * n2);
		albedo = mix(albedo, color2, smoothstep(0.6, 0.9, n1) * wear);
		rough = mix(0.78, 0.35, polish);
	} else if (kind == 10) {
		// FABRIC: a weave, and velvet's sheen at the edges
		albedo = color * (0.88 + 0.18 * n1);
		relief += sin(cpos.x * 40.0) * sin(cpos.z * 40.0 + cpos.y * 40.0) * 0.00004;
		rough = 0.96;
		rim = polish * 0.8;
		spec = 0.25;
	} else if (kind == 11) {
		// BONE: pores, and the yellow of age
		albedo = mix(color, color2, fbm(opos * 0.4) * wear);
		relief -= step(0.85, vnoise(opos * 12.0)) * 0.00008;
		rough = mix(0.7, 0.35, polish);
		sss = 0.2;
		back = albedo * 0.1;
	} else if (kind == 12) {
		// PLANT: fresh or dried (wear), a paler midrib
		albedo = mix(color, color2, wear) * (0.88 + 0.2 * n1);
		float rib = 1.0 - smoothstep(0.0, 0.035, abs(UV.x - 0.5));
		albedo = mix(albedo, albedo * 1.35, rib * pattern);
		rough = 0.68;
		sss = 0.3 * (1.0 - wear);
		back = albedo * 0.35 * (1.0 - wear);
	} else {
		// PAINTED or lacquered: a color, as glossy as its polish
		albedo = color * (0.95 + 0.08 * n1);
		rough = mix(0.85, 0.12, polish);
		coat = polish * 0.6;
	}
	// THE PLAY OF LIGHT (Props.PLAYS). Where the flames are is the renderer's business, so what
	// catches them is made of surface: a patch of layers or a flake is a small mirror facing its own
	// way, colored as a metal is, which a flame or the room's reflection lights - or does not
	vec3 nrm = NORMAL;
	vec3 tilt = vec3(0.0);      // the mirrors' facing, a nudge to the normal (the thing's space)
	float mirror = 0.0;         // how much of the surface here is one
	float aniso = 0.0;
	if (play == 1) {
		// SILK: fibers spread the light into a band across them, which slides as it turns; its
		// stripes are the banding above, laid across the fibers
		if (kind != 4 && kind != 6) {
			albedo = mix(albedo, color2, smoothstep(0.35, 0.65, sin(banding(n1)) * 0.5 + 0.5) * pattern);
		}
		// the fibers' light is spread broad across them whatever the polish (the coat keeps the
		// polish), and takes the stone's color
		aniso = 0.9;
		rough = 0.42;
		metal = max(metal, 0.5);
		spec = 0.8;
	} else if (play == 2) {
		// FLASH: layers inside lying in broad, streaky patches, tipped as they lie - a facing that
		// wanders slowly across the stone, so a whole patch catches a light at once - and throwing
		// back a color near `color2`, dark until it faces one
		vec3 q = lpos / max(grain, 0.1) + vec3(tag * 19.0);
		mirror = smoothstep(0.44, 0.62, fbm(q * vec3(0.35, 0.9, 0.35)));
		tilt = (vec3(fbm(q * 0.3 + vec3(3.1)), fbm(q * 0.3 + vec3(7.7)), fbm(q * 0.3 + vec3(13.3))) - 0.5) * 1.4;
		vec3 hsv = rgb2hsv(color2);
		float h = fract(hsv.x + (fbm(q * 0.5 + vec3(21.0)) - 0.5) * 0.18 - (1.0 - dot(NORMAL, VIEW)) * 0.06);
		albedo = mix(albedo, hsv2rgb(vec3(h, max(hsv.y, 0.75), max(hsv.z, 0.85))), mirror);
		rough = mix(rough, 0.32, mirror);
	} else if (play == 3) {
		// FIRE: small patches of every color side by side, the color shifting as it turns - seen in
		// any light, and brightest where a patch catches a flame
		vec3 c = cellsid(lpos / max(grain * 0.35, 0.05) + vec3(tag * 23.0));
		float colored = step(0.25, c.z);
		tilt = tipped(c.z, 0.25);
		float h = fract(c.z * 7.13 + (1.0 - dot(NORMAL, VIEW)) * 0.6);
		albedo = mix(albedo, hsv2rgb(vec3(h, 0.8, 1.0)), colored * 0.75);
		rough = mix(rough, 0.18, colored);
		mirror = colored * 0.6;
	} else if (play == 4) {
		// GLITTER: tiny flat flakes of `color2` all through it, each facing its own way, so a few at
		// a time catch a flame
		vec3 c = cellsid(cpos * 7.5 + vec3(seed * 3.0));
		mirror = (1.0 - smoothstep(0.26, 0.36, c.x)) * step(0.45, c.z);
		tilt = tipped(c.z, 0.65);
		albedo = mix(albedo, color2 * 1.25, mirror);
		rough = mix(rough, 0.07, mirror);
	} else if (play == 5) {
		// RAINBOW: a thin film over it, colored by its thickness and the angle it is seen at - in
		// what it reflects, as a film colors light, so a black stone stays black but for its sheen;
		// a pale body keeps its own light under it (a cream murex shell, as metal, reflected a dim
		// room and read nearly black)
		float pale = smoothstep(0.25, 0.8, dot(color, vec3(0.299, 0.587, 0.114)));
		albedo = mix(albedo, film(NORMAL, VIEW, color) * mix(0.6, 1.0, dot(color, vec3(0.333))), mix(0.8, 0.45, pale));
		metal = max(metal, mix(0.85, 0.3, pale));
		rough = clamp(rough, 0.12, 0.3);
	} else if (play == 6) {
		// GLOW: a soft light of `color2` under the surface, strongest where it is seen straight on -
		// so it floats as it turns
		float billow = smoothstep(0.3, 0.8, fbm(lpos / max(grain, 0.05) * 0.5 + vec3(11.0)));
		float g = pow(clamp(dot(NORMAL, VIEW), 0.0, 1.0), 2.5) * (0.35 + 0.65 * billow);
		albedo = mix(albedo, color2 * 1.15, g * 0.75);
		sss = max(sss, 0.55);
	}
	if (mirror > 0.0) {
		vec3 tv = mat3(VIEW_MATRIX) * (mat3(MODEL_MATRIX) * tilt);
		tv = length(tv) > 1e-6 ? normalize(tv) * length(tilt) : vec3(0.0);
		nrm = normalize(mix(nrm, normalize(nrm + tv), mirror));
		metal = mix(metal, 1.0, mirror);
	}
	// paint or inlay where the ornament is
	if (orn_paint > 0.0) {
		float m = o.y * orn_paint;
		albedo = mix(albedo, orn_color, m);
		metal = mix(metal, kind == 1 ? 0.3 : 0.0, m);
		rough = mix(rough, 0.45, m);
	}
	// few real surfaces reflect more than this much of the light: a white wax at its full value
	// blew out beside a flame
	ALBEDO = albedo * 0.84;
	METALLIC = metal;
	ROUGHNESS = rough;
	SPECULAR = spec;
	nrm = bump(nrm, VERTEX, relief);
	NORMAL = nrm;
	// THE FRAME THE LIGHT READS FIBERS IN: a silky stone's fibers spread its light ACROSS them - the
	// flow turns the frame a quarter round, so the broad axis lies across the fibers (a plain
	// surface's frame does not matter, but it must be a true one: the meshes carry no tangents)
	vec3 fv = mat3(VIEW_MATRIX) * (mat3(MODEL_MATRIX) * fibers());
	vec3 tg = fv - nrm * dot(nrm, fv);
	if (length(tg) < 1e-4) {
		tg = cross(nrm, abs(nrm.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0));
	}
	TANGENT = normalize(tg);
	BINORMAL = normalize(cross(nrm, TANGENT));
	ANISOTROPY = aniso;
	ANISOTROPY_FLOW = vec2(0.0, 1.0);
	EMISSION = emit;
	SSS_STRENGTH = sss;
	BACKLIGHT = back;
	CLEARCOAT = coat;
	CLEARCOAT_ROUGHNESS = coat_rough;
	RIM = rim;
	RIM_TINT = 0.6;
}
