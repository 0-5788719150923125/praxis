shader_type canvas_item;

// Stage filter - the whole show put through a look, in one pass.
//
// This runs on main's stage view: the single TextureRect that presents the SubViewport every
// scene draws into (see [Filters], which owns the registry and hangs this material here). So
// TEXTURE is the finished frame - every scene, every layer, the vehicle's comic page and all -
// and nothing above it on a CanvasLayer is touched, which is what keeps the karaoke line and
// the panels readable.
//
// SIX FILTERS, ONE FRAGMENT, IN A FIXED ORDER, each with its own 0..1 amount and each an exact
// no-op at 0. The order is the one in Filters.REGISTRY and it is the only part of the pipeline
// that cannot be read off this file:
//
//   1. POINTILLISM re-samples the picture into dots, so it must come first - everything below
//      then grades the dots rather than the photograph underneath them.
//   2. BLOOM is what a lens does, so it goes where a lens is: before the grade.
//   3. MONOCHROME, then 4. NOIR. Noir grades whatever it is handed, so monochrome above it is
//      black-and-white noir and monochrome below it would simply delete the split-tone.
//   5. GRAIN is the emulsion the graded light is exposed onto.
//   6. VIGNETTE is the frame, and it is last because it darkens everything including the grain.
//
// COST. Measured in AUDIT.md, ghost's GPU sits at 0.2-0.9 ms while a frame costs 30-170 ms of
// GDScript - the machine is CPU-bound by a wide margin, and a full-frame fragment shader is
// paid out of the idle part. The only filter with real fill cost is POINTILLISM (nine taps for
// the neighbourhood, see below) and it is still a fraction of one scene's `_draw`.
//
// NOTHING HERE READS THE AUDIO, deliberately. A look is a decision about the photograph; a look
// that pumped on the beat would be a scene effect, and ghost has a registry for those.

// The frame's size in PIXELS. The dot lattice and the grain are sized in device pixels rather
// than in UV, or the dots get coarser as the window gets bigger and a render at a different
// resolution is a different picture.
uniform vec2 u_size = vec2(1920.0, 1080.0);

uniform float u_pointillism : hint_range(0.0, 1.0) = 0.0;
uniform float u_bloom : hint_range(0.0, 1.0) = 0.0;
uniform float u_monochrome : hint_range(0.0, 1.0) = 0.0;
uniform float u_noir : hint_range(0.0, 1.0) = 0.0;
uniform float u_grain : hint_range(0.0, 1.0) = 0.0;
uniform float u_vignette : hint_range(0.0, 1.0) = 0.0;

// Rec. 709 luma. NOT the channel average: the eye is about six times more sensitive to green
// than to blue, so an average turns a saturated red and a saturated green of the same
// brightness into the same grey and flattens exactly the contrast a monochrome pass is for.
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

// Grain and the gate both re-roll at a FILM's rate, not at whatever the machine manages.
// Without this the noise shimmers faster on a fast frame and slower on a slow one, which is
// the single clearest tell that it is not film.
const float FILM_FPS = 24.0;
// Where on the Grain dial the transport starts misbehaving, and how often it does at the top.
// Below the onset the look is a clean print with grain on it, which is what most of the range
// should be.
const float GATE_ONSET = 0.55;
const float GATE_MAX = 0.22;
// Where on the Vignette dial the falloff stops being a lens and starts closing on all four
// edges. Below this it is photographic; above it, it is a porthole.
const float CLOSE_ONSET = 0.45;
// How heavy the grain field's tail is. Higher is sparser: the exponent is applied to a signed
// noise centred on zero, so it pushes the bulk toward nothing and leaves the extremes alone.
const float GRAIN_TAIL = 3.4;
// ...and what a surviving grain is worth. This is the amplitude of the FEW, not of the many,
// and it is small because grain MODULATES the picture - it does not punch through it. At 2.3
// the specks clipped to pure white and black and the render read as a blizzard rather than as
// stock; a strong grain should move the luminance it sits on by a fifth or so, not replace it.
const float GRAIN_GAIN = 0.30;

float luma(vec3 c) {
	return dot(c, LUMA);
}

// A cheap deterministic hash - and NOT the `fract(sin(dot(...)))` one everybody reaches for.
//
// THE SINE HASH WEAVES. Rendered and looked at, grain built on it drew a plain diagonal
// crosshatch across the whole frame: sin() is smooth, so nearby inputs give nearby outputs
// along the direction of the dot product's gradient, and on a field sampled every pixel or two
// that correlation IS the visible texture. It also loses precision on large inputs, which is
// why the frame counter feeding it has to be wrapped.
//
// This is the integer-mixing kind (Hoskins): no transcendental, no preferred direction, and
// the same cost. Everything in this file that needed randomness got better when it changed,
// not only the grain.
//
// TEXTURE IS ONLY BOUND INSIDE fragment(), so the two filters that re-sample the picture take
// the sampler as a PARAMETER rather than reaching for it. That is already the pattern here (see
// the crystal effect) and it costs one harmless engine line at compile time:
// `Condition "!actions.custom_samplers.has(...)" is true. Continuing.` - not a bug, and noted in
// CLAUDE.md as the one such line to ignore.
float hash1(vec2 p) {
	vec3 q = fract(vec3(p.xyx) * 0.1031);
	q += dot(q, q.yzx + 33.33);
	return fract((q.x + q.y) * q.z);
}

vec2 hash2(vec2 p) {
	return vec2(hash1(p), hash1(p + vec2(37.2, 11.7)));
}

// Smooth value noise on an integer lattice. Needed for the grain's MOTTLE, which is a
// low-frequency variation in density - a hash sampled coarsely and left unsmoothed is visible
// blocks, which is the one thing the mottle must not be.
float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash1(i), hash1(i + vec2(1.0, 0.0)), f.x),
		mix(hash1(i + vec2(0.0, 1.0)), hash1(i + vec2(1.0, 1.0)), f.x), f.y);
}

// THE GATE SLIP: a band of the picture displaced sideways for ONE frame, then back.
//
// Reported as part of the grain - "entire regions of the scene might fracture/shift briefly
// before correction" - and it is the other half of what reads as film. It is not an emulsion
// property at all but a TRANSPORT one: a frame that does not seat squarely in the gate is
// printed offset until the mechanism pulls it back. Nothing about noise can produce it,
// because it is a change of WHERE the picture is sampled rather than of what comes back.
//
// It happens to the SAMPLING POSITION, so it runs before anything reads the texture, and only
// the top of the dial ever slips at all: a little grain is a clean print with grain on it.
vec2 gate_slip(vec2 uv, float amount) {
	float chance = max(amount - GATE_ONSET, 0.0) / max(1.0 - GATE_ONSET, 0.001) * GATE_MAX;
	if (chance <= 0.0) {
		return uv;
	}
	float t = floor(TIME * FILM_FPS);
	if (hash1(vec2(t, 3.7)) > chance) {
		return uv;
	}
	// A band, not the whole frame: the film buckles across part of the gate.
	float y0 = hash1(vec2(t, 11.3));
	float h = mix(0.05, 0.40, hash1(vec2(t, 19.1)));
	if (uv.y < y0 || uv.y > y0 + h) {
		return uv;
	}
	float shift = (hash1(vec2(t, 23.7)) - 0.5) * mix(3.0, 30.0, amount) / max(u_size.x, 1.0);
	return vec2(clamp(uv.x + shift, 0.0, 1.0), uv.y);
}

// --- 1. POINTILLISM ----------------------------------------------------------------------
//
// The picture re-laid as overlapping dots of paint. A regular lattice of discs is a halftone
// screen and reads as print, so the cell centres are JITTERED and the dots are allowed to
// overlap; what makes it read as paint rather than as a dotted grid is that each dot takes its
// colour from its own centre and its RADIUS from how bright that is.
//
// THE NEIGHBOURHOOD IS THE WHOLE TRICK. A pixel belongs to one cell, but at this dot size the
// dot covering it is very often the one from the cell NEXT DOOR - a version that only asked its
// own cell leaves a hard rectangular gap around every dot and reads as a grid of stamps. So the
// 3x3 cells around the pixel are tested and the nearest covering dot wins. Nine taps, which is
// this filter's entire cost.
//
// The ground between the dots is the containing cell's own colour, well dimmed: pointillism on
// white paper looks right on a bright painting and turns a night scene into a blank sheet, and
// a dark ground keeps a dark picture legible at any exposure.
vec3 pointillism(sampler2D tex, vec2 uv, vec3 src, float amount) {
	vec2 px = uv * u_size;
	// The dot pitch, in pixels. The bottom of the range is a canvas tooth you have to look for;
	// the top is a painting you have to stand back from.
	float cell = mix(3.0, 22.0, amount);
	vec2 id = floor(px / cell);
	vec3 col = src;
	float best = 1e9;
	vec3 ground = src;
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			vec2 nid = id + vec2(float(i), float(j));
			// +-0.35 of a cell: enough that no row of dots lines up, not so much that a dot
			// leaves the cells this loop looks at.
			vec2 c = (nid + 0.5 + (hash2(nid) - 0.5) * 0.7) * cell;
			vec3 tap = texture(tex, clamp(c / u_size, vec2(0.0), vec2(1.0))).rgb;
			if (i == 0 && j == 0) {
				ground = tap * 0.22;
			}
			// A bright dab of paint is a fatter dab. The floor keeps the dark half of the
			// picture from dissolving into bare ground.
			float r = cell * 0.5 * (0.45 + 0.75 * luma(tap)) * (0.8 + 0.4 * hash1(nid + 5.0));
			float d = length(px - c) - r;
			if (d < best) {
				best = d;
				col = tap;
			}
		}
	}
	// One pixel of softness on the rim: a hard edge aliases, and paint has none.
	float cover = 1.0 - smoothstep(-1.0, 1.0, best);
	return mix(src, mix(ground, col, cover), amount);
}

// --- 2. BLOOM ----------------------------------------------------------------------------
//
// Light bleeding out of the bright parts. Only what is ALREADY above the picture's highlights
// blooms - a threshold, not a blur added to everything - because a bloom with no threshold is
// just a fogged lens, and that was the first version of this.
//
// Eight taps on a ring rather than a separable gaussian: this is a glow, not a defocus, and a
// second pass would need a second render target for a filter nobody will look at closely.
vec3 bloom(sampler2D tex, vec2 uv, vec3 src, float amount) {
	float radius = mix(2.0, 14.0, amount) / max(u_size.y, 1.0);
	vec3 sum = vec3(0.0);
	for (int i = 0; i < 8; i++) {
		float a = float(i) * 0.7853981634;   // TAU / 8
		// Two rings in one loop - the odd taps sit at twice the radius, which spreads the
		// falloff without doubling the sample count.
		float rr = radius * (float(i % 2) + 1.0);
		vec2 off = vec2(cos(a), sin(a)) * rr * vec2(u_size.y / max(u_size.x, 1.0), 1.0);
		vec3 tap = texture(tex, clamp(uv + off, vec2(0.0), vec2(1.0))).rgb;
		sum += max(tap - vec3(0.62), vec3(0.0));
	}
	return src + sum * (amount * 0.55);
}

// --- 3. MONOCHROME ------------------------------------------------------------------------
vec3 monochrome(vec3 src, float amount) {
	return mix(src, vec3(luma(src)), amount);
}

// --- 4. NOIR ------------------------------------------------------------------------------
//
// The hard grade, and it is three things rather than a contrast slider. An S-curve closes the
// blacks and burns the highlights; a little of the remaining colour is pulled out, because
// noir is not saturated even when it is not monochrome; and the shadows go COLD against warm
// highlights, which is the split-tone that reads as noir rather than as "high contrast".
vec3 noir(vec3 src, float amount) {
	float l = luma(src);
	// smoothstep IS the S-curve. Its ends move inward with the amount, so the dial is the
	// width of the tonal range that survives.
	float lo = mix(0.0, 0.34, amount);
	float hi = mix(1.0, 0.66, amount);
	float curved = smoothstep(lo, hi, l);
	// Re-apply the original chroma around the new luma rather than grading each channel, so
	// the curve cannot push a channel into a hue shift of its own.
	vec3 graded = src + vec3(curved - l);
	graded = mix(graded, vec3(curved), amount * 0.45);
	vec3 cold = vec3(0.82, 0.90, 1.12);
	vec3 warm = vec3(1.10, 1.02, 0.86);
	graded *= mix(cold, warm, curved);
	return mix(src, clamp(graded, vec3(0.0), vec3(4.0)), amount);
}

// --- 5. GRAIN ----------------------------------------------------------------------------
//
// Film emulsion. The first version of this was one hash per pixel scaled by a midtone weight,
// and it was reported as "too uniform and consistent... it doesn't look like any version of
// film grain I've ever seen". That is an accurate description of what it was: per-pixel white
// noise is TELEVISION STATIC, or a digital sensor at high ISO, and neither is film. Four
// things separate the two, and the first two are why it read wrong.
//
//   GRAIN HAS A SIZE. It is clumps of developed silver, not pixels - so the noise is sampled
//   on a lattice a few pixels across, in two octaves so the clumps have structure of their own
//   and the lattice never shows. At one pixel there is no grain to see, only a texture.
//
//   GRAIN IS SPARSE. The visible grains are scattered specks with a great deal of almost
//   nothing between them; a uniform hash puts the same amount of noise on every pixel, which
//   is exactly the "uniform and consistent" that was reported. Raising the signed noise to a
//   power gives the heavy tail a real grain field has: most of the frame carries nearly
//   nothing and the few grains that land, land hard.
//
//   IT IS UNEVENLY DENSE. Real stock mottles - regions grainier than their neighbours,
//   drifting slowly. Without it even sparse grain reads as an even wash at a distance.
//
//   ...AND IT IS IN THE MIDTONES. Silver that did and did not develop: none in a black that
//   exposed nothing, none in a highlight that exposed completely. A flat noise floor fogs the
//   blacks, which is the one place it must not.
//
// The dial also opens the gate slip (see gate_slip, above) past GATE_ONSET, which is the other
// half of the same report.
vec3 grain(vec2 uv, vec3 src, float amount) {
	vec2 px = uv * u_size;
	// Wrapped, because the hash below is a sine: after a few minutes an unwrapped frame
	// counter is large enough to lose the precision the hash depends on, and the noise starts
	// repeating and banding instead of re-rolling.
	float t = mod(floor(TIME * FILM_FPS), 1024.0);
	vec2 drift = vec2(t * 41.0, t * 17.0);

	// SIZE, AND NOT ON A LATTICE. Flooring a hash onto cells of grain-size is a GRID, and a
	// grid shows: rendered and looked at, the first version of this drew a plain crosshatch
	// over the whole frame, which is the "too uniform and consistent" being reported from the
	// other side. Smooth value noise gives clumps the same size with nothing to align to.
	float gsz = mix(1.6, 3.2, amount);
	float n = vnoise(px / gsz + drift) * 0.62
		+ vnoise(px / (gsz * 0.45) + drift.yx) * 0.38;

	// SPARSE. Smooth noise clusters around its middle, so signing it and raising it to a power
	// leaves most of the frame at nearly nothing and lets the few grains that do land stand
	// clear of it. That histogram - almost everything near zero, a thin heavy tail - is what
	// separates a grain field from a texture laid over the picture.
	float sn = n * 2.0 - 1.0;
	sn = sign(sn) * pow(abs(sn), GRAIN_TAIL);

	// MOTTLE. Whole REGIONS grainier than their neighbours, drifting slowly. Without it even
	// sparse grain averages out to an even wash at any distance, which is the other half of
	// "uniform and consistent". Anisotropic and in frame units, so it does not read as a
	// circular pattern and does not change scale with the window.
	float m = vnoise(uv * vec2(5.0, 3.0) + vec2(t * 0.011, t * -0.008));
	float density = 0.06 + 1.9 * smoothstep(0.34, 0.86, m);

	// ...and the exposure weight, peaking at mid-grey and vanishing at both ends.
	float l = luma(src);
	float weight = 4.0 * l * (1.0 - l);
	return src + vec3(sn * amount * GRAIN_GAIN * density * weight);
}

// --- 6. VIGNETTE -------------------------------------------------------------------------
//
// A LENS VIGNETTES IN A CIRCLE, in the image plane, so the distance is measured with the aspect
// PUT BACK - x scaled by width/height - and then normalised by the frame's own half-diagonal.
// The obvious version, `length(uv - 0.5)`, is an ellipse stretched to the frame: it makes the
// top and bottom edges fall off exactly as hard as the left and right ones, when on a widescreen
// frame the sides are nearly twice as far from the optical axis.
//
// ...AND THAT IS WHY IT COULD NOT CLOSE. Reported as "even at 1.0 it doesn't cover ANY of the
// edges - it only covers the corners", which was true and was the circle being obeyed too
// faithfully: on a 16:9 frame the middle of the top edge is 0.49 of the way to a corner, so a
// falloff deep enough to darken it has to start halfway to the centre, and then it is not a
// vignette any more, it is an iris. The two things wanted here are not the same shape.
//
// So the DIAL CROSSES BETWEEN THEM. Below CLOSE_ONSET it is a lens, exactly as before, and the
// aspect behaviour the gate measures is the point of it. Above, the distance blends toward the
// EDGE distance - which reaches 1 at the middle of every edge, not just at the corners - so the
// top of the range closes in on all four sides like a porthole, which a lens does not do and
// which is what "even at 1.0" was asking for. A look is allowed to leave physics behind at the
// end of its travel; it is not allowed to do so in the middle, where most settings live.
vec3 vignette(vec2 uv, vec3 src, float amount) {
	float aspect = max(u_size.x, 1.0) / max(u_size.y, 1.0);
	vec2 d = (uv - vec2(0.5)) * vec2(aspect, 1.0);
	float r_lens = length(d) / length(vec2(0.5 * aspect, 0.5));
	// 1 at the middle of EVERY edge, whatever the aspect.
	vec2 e = abs(uv - vec2(0.5)) * 2.0;
	float r_edge = max(e.x, e.y);
	float close = clamp((amount - CLOSE_ONSET) / max(1.0 - CLOSE_ONSET, 0.001), 0.0, 1.0);
	float r = mix(r_lens, max(r_lens, r_edge), close);
	// The falloff's INNER edge moves with the dial as well as its depth, so a little is a
	// darkened corner and a lot is a porthole, rather than one fixed shape fading in.
	float fall = 1.0 - amount * smoothstep(mix(0.90, 0.30, amount), 1.0, r);
	return src * fall;
}


void fragment() {
	// THE GATE SLIP MOVES THE SAMPLING POSITION, so it has to happen before anything reads the
	// texture - including pointillism and bloom, which take their own taps. Everything below
	// works in `uv` rather than in UV for that reason.
	vec2 uv = UV;
	if (u_grain > 0.0) {
		uv = gate_slip(uv, u_grain);
	}
	vec4 tex = texture(TEXTURE, uv);
	vec3 c = tex.rgb;
	// Each filter is skipped OUTRIGHT at zero rather than multiplied by zero. Pointillism and
	// bloom are nine and eight taps; paying for them on every frame of a show that has neither
	// switched on is the kind of cost nobody ever notices and nobody can explain later.
	if (u_pointillism > 0.0) {
		c = pointillism(TEXTURE, uv, c, u_pointillism);
	}
	if (u_bloom > 0.0) {
		c = bloom(TEXTURE, uv, c, u_bloom);
	}
	if (u_monochrome > 0.0) {
		c = monochrome(c, u_monochrome);
	}
	if (u_noir > 0.0) {
		c = noir(c, u_noir);
	}
	if (u_grain > 0.0) {
		c = grain(uv, c, u_grain);
	}
	if (u_vignette > 0.0) {
		c = vignette(uv, c, u_vignette);
	}
	COLOR = vec4(clamp(c, vec3(0.0), vec3(1.0)), tex.a);
}
