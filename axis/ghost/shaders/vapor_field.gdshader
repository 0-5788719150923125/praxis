shader_type canvas_item;
render_mode blend_add;

// Vapor field - the volumetric body of [Layer.Vapor].
//
// WHAT IT IS. Coloured vapour: heavy, twisting masses of ink-in-water with torn, wispy
// fronts and threads of stretched cotton pulling off them, lit from inside. Every pixel of
// one quad evaluates the same density field; the structure is in the arithmetic, not in a
// crowd of sprites.
//
// WHY IT IS A FIELD AT ALL. ghost's existing atmosphere - `clouds`, `fog_bank`,
// `fog_volume`, `aurora` - is stacked gaussian puffs, and a stack of puffs can only ever be
// SOFT: no front, no interior, one hue per sheet. The look this file is for is diffuse but
// HARD, with a boundary you can see, filaments inside the mass, and colour that changes
// across it. None of that is a blob's to draw.
//
// THE GENERATOR IS A RIDGED MULTIFRACTAL, and that choice is the whole file. Two earlier
// cuts built the density from ordinary fbm and then tried to add filaments as a separate
// anisotropic ridged noise sampled in the flow's frame. Both read as engraved satin - long
// parallel curves over the whole frame, like a contour map - because that is what ridged
// noise squashed across a smooth direction field IS. Rendered, looked at, twice.
//   A ridged multifractal has the filaments in the DENSITY: each octave is creased
//   (1 - |2n-1|, raised to a power), and each octave's amplitude is weighted by the octave
//   above it, so detail collects ON the creases and nowhere else. That is the standard
//   generator for cirrus, smoke and nebula wisps, it is isotropic (so it cannot band), and
//   its level sets are exactly the torn edge this look needs.
//
// THE OTHER THREE PARTS:
//   TWIST - the sample point is domain-warped twice by low-frequency noise that scrolls in
//   time, coarsely (the mass turning over) then finely (the curl inside it). One warp gives
//   billows; two give the folded, re-entrant shapes ink makes in water.
//   STRETCH - the multifractal is sampled in a frame aligned to a smooth flow ANGLE and
//   mildly squashed across it, which elongates the filaments along the flow. MILDLY: this is
//   the knob that turns cotton into corduroy if it is pushed, and the direction comes from
//   an angle field rather than a normalized vector because normalizing is degenerate where
//   the vector passes through zero - which showed as a fan of rays converging on that point.
//   LIGHT - the plume envelope is analytic, so it can be re-evaluated a couple of steps
//   toward the lamp and used as optical depth: exp(-k * occlusion). The far side of a mass
//   goes dark and its near side glows, which is why it reads as volume and not as a gradient.
//
// COLOUR is per PLUME, mixed by contribution, so a violet mass and a teal one blend where
// they overlap and stay themselves where they do not; and it has DEPTH - the fringe takes
// the scheme's second hue, the core keeps its own, the hottest creases bleach to white.
//
// Time is a uniform (the layer's clock), never TIME - live and the offline bake step the
// same clock, so a seed reproduces its vapour. Same rule as flame.gdshader.

// The plumes: centre, amplitude, elliptical radii, colour. Each is driven by its own
// harmonic band on the CPU side (see Layer.Vapor).
uniform int u_n = 5;
uniform float u_px[8];
uniform float u_py[8];
uniform float u_amp[8];
uniform float u_rx[8];
uniform float u_ry[8];
uniform vec3 u_col[8];
// Per-plume GLOW: a beat's answer. It scales that plume's LIGHT and touches nothing
// geometric, which is the whole discipline of this file's audio coupling - see the note on
// u_burst below.
uniform float u_glow[8];

// AN INTEGRATED PHASE, not a clock. The layer accumulates it (phase += dt * rate), and the
// rate is where the music reaches the motion. It used to be `u_time * u_churn` with a
// live-modulated churn, which is a bug and not a small one: multiplying an absolute clock by a
// changing rate JUMPS the phase by (dt_rate x t), so at a hundred seconds in, a rate nudge
// teleported the entire field. Reported as the scene being "EXTREMELY unstable".
uniform float u_time = 0.0;
// The medium's own steady travel, integrated by the layer the same way. This is the
// "continuous movement forward": the vapour streams through the frame at a constant rate
// whatever the music does.
uniform vec2 u_drift = vec2(0.0);
uniform vec2 u_half = vec2(0.9, 0.55);   // quad half-extents in unit fractions
uniform vec2 u_seed = vec2(0.0);         // noise offset       (sampled per instance)
uniform float u_scale = 1.0;             // feature size       (sampled per instance)
uniform float u_swirl = 0.6;             // domain-warp gain: how hard it twists
uniform float u_crease = 2.0;            // ridge sharpness: 1 = billowy, 3 = wiry filament
uniform float u_stretch = 2.0;           // flow-aligned anisotropy (1 = none, 3.5 = drawn out)
uniform float u_hard = 0.6;              // 0 = haze-soft front, 1 = a cut front
uniform float u_thresh = 0.5;            // density the front sits at
uniform float u_haze = 0.16;             // unthresholded bloom around the masses
uniform float u_gain = 1.0;              // brightness         (sampled per instance)
// Beat surge - BRIGHTNESS ONLY, and that restriction is the point. It used to also widen the
// density and push the warp gain, and both of those move geometry: a mass inflating on the beat
// and space re-warping under it read exactly as reported - "it expands and contracts rapidly
// with the harmonics... it should be continuous movement forward, not an expand and contract".
// Light may pulse. Shape may not.
uniform float u_burst = 0.0;
uniform vec2 u_light = vec2(-0.45, -0.25); // the lamp, in the same unit fractions
uniform float u_shadow = 2.0;            // optical depth per unit of plume mass
uniform vec3 u_rim = vec3(0.35, 0.65, 1.0); // the fringe chroma (the scheme's accent)

// The same mod-289 lattice hash flame.gdshader uses, and for the same reason: inputs are
// bounded BEFORE any multiply, so precision holds however far the time scroll pushes the
// coordinates. A fract-of-big-product hash quantizes within seconds of scrolling and the
// interior tiles into blocks. (Copied rather than shared through an include: that would
// mean editing flame.gdshader, which a running render depends on.)
float permf(float x) {
	return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float hash21(vec2 p) {
	p = p - floor(p / 289.0) * 289.0;
	return fract(permf(permf(p.x) + p.y) * 0.145598);
}

float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float a = hash21(i);
	float b = hash21(i + vec2(1.0, 0.0));
	float c = hash21(i + vec2(0.0, 1.0));
	float d = hash21(i + vec2(1.0, 1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Three octaves for the WARPS - they are steered, not looked at, and detail there costs a
// lot while showing as nothing.
float fbm3(vec2 p) {
	float v = 0.0;
	float a = 0.5;
	for (int i = 0; i < 3; i++) {
		v += a * vnoise(p);
		p = mat2(vec2(1.6, 1.2), vec2(-1.2, 1.6)) * p;
		a *= 0.5;
	}
	return v;
}

// THE BODY: a ridged multifractal. `sharp` creases each octave and the running weight `w`
// admits the next octave only where this one is already high, so fine detail lives on the
// filaments and the space between them stays smooth. Returns roughly 0..1.
float ridged(vec2 p, float sharp) {
	float sum = 0.0;
	float amp = 0.55;
	float w = 1.0;
	for (int i = 0; i < 5; i++) {
		float n = 1.0 - abs(2.0 * vnoise(p) - 1.0);
		n = pow(n, sharp);
		sum += n * amp * w;
		w = clamp(n * 1.6, 0.0, 1.0);
		p = mat2(vec2(1.6, 1.2), vec2(-1.2, 1.6)) * p;   // determinant 4: an octave per step
		amp *= 0.5;
	}
	return sum;
}

// The plume envelope: a sum of elliptical gaussians. Analytic and cheap, which is what lets
// it be tapped again along the light ray below.
float env(vec2 p) {
	float m = 0.0;
	for (int i = 0; i < u_n; i++) {
		vec2 d = vec2((p.x - u_px[i]) / u_rx[i], (p.y - u_py[i]) / u_ry[i]);
		m += u_amp[i] * exp(-dot(d, d));
	}
	return min(m, 1.35);
}

// The same envelope, also returning the colour the plumes mix to here - weighted by
// contribution, so an overlap blends and an isolated mass keeps its own hue.
float env_col(vec2 p, out vec3 chroma, out float glow) {
	float m = 0.0;
	vec3 acc = vec3(0.0);
	float gl = 0.0;
	for (int i = 0; i < u_n; i++) {
		vec2 d = vec2((p.x - u_px[i]) / u_rx[i], (p.y - u_py[i]) / u_ry[i]);
		float g = u_amp[i] * exp(-dot(d, d));
		m += g;
		acc += u_col[i] * g;
		gl += u_glow[i] * g;
	}
	chroma = acc / max(m, 1e-4);
	glow = gl / max(m, 1e-4);
	// CAPPED, like the fire bed's env: three overlapping plumes at full amplitude sum past
	// 3, and everything downstream of density saturates - the frame becomes one white shape
	// with wisps round the outside. The cap is what keeps an overlap reading as a denser
	// mass rather than as a blowout.
	return min(m, 1.35);
}

void fragment() {
	vec4 fade = COLOR;  // white quad texture x modulate: the scene's fade rides in here

	vec2 p = (UV - 0.5) * 2.0 * u_half;   // unit fractions, y down, (0,0) at centre
	float tt = u_time;      // already a phase (see the uniform)

	// --- TWIST: two domain warps, both scrolling ---
	vec2 q = p * (1.5 * u_scale) + u_seed;
	vec2 w1 = vec2(fbm3(q * 0.75 + vec2(0.0, -tt * 0.06)),
			fbm3(q * 0.75 + vec2(tt * 0.05, 0.0) + 37.1)) - 0.5;
	vec2 pw = p + w1 * u_swirl;
	vec2 w2 = vec2(fbm3(pw * (3.0 * u_scale) + vec2(tt * 0.10, 0.0) + 11.3),
			fbm3(pw * (3.0 * u_scale) - vec2(0.0, tt * 0.12) + 71.9)) - 0.5;
	// The fine warp is weighted generously: it is also what stops the PLUME ELLIPSES from
	// showing. A gaussian envelope times a locally smooth multifractal has circular level
	// sets, and those render as concentric onion rings around a mass (rendered, looked at -
	// they were there). Distorting the envelope's own input breaks them into an organic edge.
	vec2 pw2 = pw + w2 * u_swirl * 0.55;

	// --- STRETCH: the flow frame, from a smooth angle field ---
	float ang = 6.28318 * fbm3(pw * (1.1 * u_scale) + u_seed * 0.11 + 5.3);
	vec2 dir = vec2(cos(ang), sin(ang));
	vec2 per = vec2(-dir.y, dir.x);
	// The MEDIUM travels, its envelope does not: the drift is added for the noise sampling
	// only, so the filaments stream steadily through masses that stay where the composition
	// put them. (Drifting the envelope too would walk every mass off the frame and empty it.)
	vec2 pd = pw2 + u_drift;
	vec2 pf = vec2(dot(pd, dir), dot(pd, per) * u_stretch);

	// --- the mass, its colour, and how deep inside it we are ---
	vec3 chroma;
	float plume_glow;
	float m = env_col(pw2, chroma, plume_glow);
	float nb = ridged(pf * (2.7 * u_scale) + u_seed * 0.37, u_crease);
	float d = m * (0.16 + 1.55 * nb);
	// u_hard is THREE things, because "hard" is not one number: the width of the front, how
	// fast the interior brightens past it, and how much diffuse haze surrounds the mass. Only
	// the first was wired at first and the knob measured as barely doing anything (x1.14 on
	// the gate's edge-step, where the whole point of the scene is being harder than cloud) -
	// a narrow smoothstep on the body is invisible when the haze around it is what the eye
	// sees at the boundary anyway.
	float hardk = clamp(u_hard, 0.0, 1.0);
	float soft = mix(0.34, 0.030, hardk);
	float body = smoothstep(u_thresh - soft, u_thresh + soft, d);
	// THICKNESS - how far past the front, normalized. This is what gives a mass an
	// INTERIOR: brightness and colour both ride it, so the vapour reads as something with
	// depth rather than a flat shape with texture on it.
	// The divisor is generous on purpose. Capped mass times a near-1 multifractal puts the
	// whole core of a mass past a tight divisor, so thickness saturates over a wide region
	// and the interior renders as one flat pale plateau (it did). Normalizing over the full
	// range instead keeps the interior a GRADIENT all the way in.
	float thick = clamp((d - u_thresh) / 1.10, 0.0, 1.0);

	// --- LIGHT ---
	vec2 lv = u_light - p;
	vec2 ldir = normalize(lv + vec2(1e-5, 0.0));
	float occ = env(pw2 + ldir * 0.10) * 0.62 + env(pw2 + ldir * 0.24) * 0.38;
	float lit = exp(-u_shadow * occ);
	float lamp = 1.0 / (1.0 + 1.9 * dot(lv, lv));         // the lamp's own falloff
	float shade = 0.26 + 1.05 * lit * (0.45 + 0.80 * lamp);
	// A filament lying crossways to the lamp catches it, the way real fibre does. `nb` is
	// high exactly on the creases, so this rides the threads and not the mass.
	float sheen = pow(1.0 - abs(dot(dir, ldir)), 4.0) * smoothstep(0.45, 0.95, nb) * lit;

	// --- COLOUR: chromatic DEPTH ---
	// The fringe of a mass is not a dimmer version of its middle: a thin edge takes the
	// scheme's second hue and the core keeps its own, so one mass grades across the wheel
	// the way ink in water does, and the hottest creases bleach toward white.
	vec3 edge = mix(chroma, u_rim, 0.60);
	vec3 col = mix(edge, chroma, smoothstep(0.0, 0.55, thick));
	float hot = smoothstep(0.68, 1.15, thick * (0.5 + 0.9 * shade));
	col = mix(col, mix(chroma, vec3(1.0), 0.45), hot);
	col += u_rim * sheen * 0.55;

	// The emission: the lit body, brightening with thickness and creased by the same field
	// that cuts its edge, plus an unthresholded HAZE so the masses sit in their own glow
	// instead of being cut out of black. The haze carries most of the frame's diffuse colour
	// and is what keeps a hard front from reading as a decal.
	float glow = body * (0.26 + 0.80 * pow(thick, mix(1.15, 0.50, hardk))) * (0.55 + 0.85 * nb);
	// THE FRONT GLINT. A band of extra light right at the boundary, and only when the mass is
	// a hard one - the leading edge of lit vapour really is brighter than its interior
	// (backscatter), and it is what makes a front read AS a front rather than as the place the
	// gradient happens to end. Modest on purpose: pushed further it starts drawing every
	// fractal wobble of the level set as its own bright line, which is the engraved-satin
	// failure this file spent two rewrites getting away from.
	float band = exp(-pow((d - u_thresh) / mix(0.30, 0.075, hardk), 2.0));
	glow += 0.42 * hardk * band * (0.30 + 0.70 * lit) * smoothstep(0.05, 0.45, m);
	vec3 emit = col * glow * shade
		+ chroma * (u_haze * m * (0.35 + 0.75 * nb) * (0.40 + 0.60 * lit) * mix(2.1, 1.15, hardk));
	// The music's whole claim on the picture: brightness. A beat lifts the plume it landed on
	// (plume_glow, localized) and the transient lifts the frame a little (u_burst), and neither
	// moves anything.
	emit *= u_gain * (1.0 + 0.30 * u_burst) * (1.0 + 0.55 * plume_glow);
	// A SOFT SHOULDER, ON LUMINANCE. Additive light over a colour bed has no ceiling, and
	// without this the densest part of a mass clips flat over a good fraction of the frame
	// (rendered, looked at - it did). Compressing PER CHANNEL fixes the clipping but washes
	// the core toward pale grey, because the channel already near the ceiling is compressed
	// hardest and the ratios between them collapse - which is the same plateau one step
	// desaturated. Scaling all three by one luminance-derived factor keeps the hue and the
	// saturation exactly and only takes the level down, so a hot core stays hot MAGENTA.
	float lum = dot(emit, vec3(0.299, 0.587, 0.114));
	emit *= 1.0 / (1.0 + 0.55 * lum);
	// ... and a small saturation push AFTER it. Compression plus additive overlap still walks
	// the brightest masses toward cream; a scene whose subject is colour should end up more
	// saturated than a photograph, not less.
	float lum2 = dot(emit, vec3(0.299, 0.587, 0.114));
	emit = mix(vec3(lum2), emit, 1.28);

	// Alpha 1 with blend_add (src.rgb * src.a + dst): the emission adds as computed and the
	// scene's fade - which rides in COLOR.a - scales it linearly.
	COLOR = vec4(emit, 1.0) * fade;
}
