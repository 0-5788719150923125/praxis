shader_type canvas_item;
// blend_disabled: this shader WRITES A FIELD, not a picture - all four
// channels are raw state. Under the default blend_mix the output RGB was
// multiplied by the coat's own alpha on write, and outside the face the
// never-cleared buffer's garbage alpha survived every step (the whole-frame
// coat bug).
render_mode blend_disabled;

// The clown's LIQUID PAINT field - a persistent, advected simulation the
// detector only DEPOSITS into. Runs in a ping-pong SubViewport pair (see
// MaskEditor._step_paint_sim); mask_split.gdshader samples the result as
// u_clown_paint instead of drawing feature shapes analytically.
//
// Channels: R = eye black (both patches), G = lip red, B = nose red,
// A = the white coat's face region (screen-space - the coat has NO fitted
// oval geometry left to rotate or rescale with anything else).
//
// Per step, per texel:
//   DEPOSIT - the current model's feature ellipses write in at full value
//             immediately (paint arrives fast where the feature IS);
//   ADVECT  - existing paint drifts toward its channel's attractor (the
//             feature's current position), so when a re-fit jumps, the
//             orphaned residue physically FLOWS into the new spot instead
//             of fading in place - the "colors morph into their new
//             position" behavior this exists for;
//   OOZE    - a slow noise wander + slight diffusion keeps the blobs
//             organic, liquid, never die-cut;
//   DECAY   - whatever never reaches a feature thins away (~0.5s).
//
// Like META's mirror, this is a bounded one-step-per-frame feedback loop -
// frame-rate coupled by nature, stepped on PLAYBACK time deltas so pause
// freezes the liquid and export traces the same currents.

uniform sampler2D u_prev : hint_default_black;
// The live frame + the face's own colour statistics: the deposits below are
// EVIDENCE-DRIVEN, not stamped shapes. A landmark only says WHERE TO LOOK;
// the pixels say what the feature's shape is (dark socket, red lip line, lit
// nose ridge), the same way the white coat is a per-pixel face-tint match
// rather than a fitted oval. Stamped ellipses were the "circles drawn over
// the face" complaint.
uniform sampler2D u_frame : hint_default_black;
uniform float u_face_lum = 0.4;    // mean luminance over the detected face
uniform float u_face_red = 0.02;   // mean (r - g) over it - lips are redder than THIS
uniform float u_dt = 0.016;        // playback seconds advanced this step
uniform int u_reset = 0;           // 1 = snap the field to the targets (seek/first frame)
uniform float u_time = 0.0;        // playback clock, for the ooze wander
uniform float u_aspect = 1.7778;
uniform vec2 u_eye_l = vec2(0.42, 0.40);
uniform vec2 u_eye_r = vec2(0.58, 0.40);
uniform vec2 u_mouth = vec2(0.5, 0.62);
uniform vec2 u_nose = vec2(0.5, 0.52);       // its own slow anchor, not re-derived here
uniform vec2 u_face_c = vec2(0.5, 0.45);     // face centroid - the coat channel's home
uniform vec2 u_face_r = vec2(0.14, 0.20);    // face semi-axes (uv units)
// Feature sizes in PLAIN SCREEN units (uv height). Deliberately NOT
// eye-distance units: normalizing by the eye pair re-coupled every feature's
// size to the pair's jitter, and the whole mask rescaled in lockstep.
uniform float u_eye_lr = 0.035;
uniform float u_eye_rr = 0.035;
uniform vec2 u_mouth_r = vec2(0.05, 0.028);  // x in uv-x units, y in uv-y units
uniform float u_scale = 1.0;       // the clown layer's Scale knob (feature sizes)

// PLAUSIBILITY CAPS ON THE FITTED WIDTHS, in HEIGHT units (the space this whole
// shader works in - see `p = UV * asp` below), never in raw uv. The distinction
// is the difference between working and not on anything that isn't 16:9: a cap
// of "0.145 of the frame's width" is 0.258 frame-heights on a landscape clip and
// 0.082 on a portrait one, so the identical rule that lets a face through on one
// crushes it to a third of its width on the other. These are the original raw-uv
// bounds converted at 16:9, so landscape footage is unchanged to the bit and
// every other shape now gets the same PHYSICAL cap rather than a scaled one.
const float FACE_CAP_XLO = 0.0889;    // was clamp(u_face_r.x, 0.05, ...) in raw uv
const float FACE_CAP_XHI = 0.2578;    // was clamp(u_face_r.x, ..., 0.145)
const float MOUTH_CAP_XLO = 0.0267;   // was clamp(u_mouth_r.x, 0.015, ...)
const float MOUTH_CAP_XHI = 0.0978;   // was clamp(u_mouth_r.x, ..., 0.055)
// How fast the eye paint runs downward, in aspect-space units per second, against
// an attractor pull of 2.4 per unit of offset - so it settles a little under the
// socket rather than streaming off the chin. See the deposit's advection call.
const float EYE_DRIP = 0.10;
// How hard the COAT channel is pulled toward the face centre, against 2.4 for the
// features. See the deposit's advection call for why it is nearly still.
const float COAT_PULL = 0.35;
// The three clown-specific knobs, each a view onto an existing per-layer
// field (fx_smooth / fx_lag / fx_stick - see MaskSession's doc):
uniform float u_bleed = 0.0;       // Bleed  - how far the paint spreads/softens
uniform float u_settle = 0.35;     // Settle - how sticky the paint is in time
uniform float u_hollow = 0.0;      // Hollow - keep the paint AROUND a feature
// EVIDENCE - how much of a drawn feature is the PICTURE and how much is the
// stamped window it is drawn inside. Every feature here is a window (an ellipse
// for the eye patches and the nose ball, a wide shallow one for the lips)
// multiplied by what the frame actually shows there - a dark socket, a red lip
// line, a lit nose ridge. The multiply has always had a FLOOR, so a fixed share
// of each feature is the bare shape whatever the picture says, which is what
// reads as "a circle for the nose and a smile for the mouth" rather than paint
// on a nose and a mouth. Raising this lowers those floors toward nothing, so the
// windows become bounds on WHERE paint may go and the picture decides where it
// actually lands. 0 is the look this always had.
uniform float u_evidence = 0.0;
// THE FEATURE STENCIL - the landmark contours, rasterized (see MaskEditor's
// _update_stencil). R = the two eyes, G = the lips, B = the nose, A = the face
// oval, each a real outline rather than an ellipse. This is what lets a nose stop
// being a circle: the deposit windows below are now SHAPES the face actually has.
// Absent (all zero) the branch falls back to the analytic ellipses, which is what
// happens when there is no landmark track.
uniform sampler2D u_stencil : hint_default_black;
uniform float u_has_stencil = 0.0;
// How far the eye black runs DOWN the cheek, past the eye itself - the drip a
// clown's eye paint has and a symmetric blob cannot. Distinct from the small
// always-on downward drift in the advection: this one is a deliberate, sized
// tongue of paint below each eye, and it is the author's to set.
uniform float u_drip = 0.0;
// The run's WEIGHT and its BOW - the two things about a tear of liner that are a
// look rather than a measurement, so the two the author gets a say over. Width
// scales the source width the stencil walk measured (so it stays a fraction of
// THIS eye, whatever size it is); curve scales the outboard bias, from falling
// nearly straight to sweeping well out across the cheekbone.
// SMUDGE - how far the eye paint is rubbed past its own contour, and how softly
// it ends. See the edge treatment in the deposit for what it actually does. Its
// zero is a soft edge, not the polygon: a hard edge is the fault, not a setting.
uniform float u_eye_smudge = 0.45;
uniform float u_drip_w = 0.35;
uniform float u_drip_curve = 0.45;
// How far the COAT's outer edge fades into the picture, in frame UV.
//
// The coat's silhouette is a rasterized polygon, so its alpha steps from 1 to 0
// across about one pixel. That did not show while the sim's decay and drift were
// softening the boundary - but flooring the coat to its deposit (which is what
// stopped the paint falling short of the jaw) also made the boundary exactly as
// crisp as the polygon, and a hard-edged white shape sitting slightly proud of a
// face reads as a sticker. There is no gradient in a 1px edge to shape, so the
// feather has to be MADE: the alpha is averaged over a disc of this radius, which
// turns the step into a ramp of that width.
uniform float u_coat_feather = 0.012;

float permf(float x) {
	return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float hash21p(vec2 p) {
	p = p - floor(p / 289.0) * 289.0;
	return fract(permf(permf(p.x) + p.y) * 0.145598);
}

float vnoisep(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float a = hash21p(i);
	float b = hash21p(i + vec2(1.0, 0.0));
	float c = hash21p(i + vec2(0.0, 1.0));
	float d = hash21p(i + vec2(1.0, 1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Distance from a point to a SEGMENT. The eye-drip marches a centreline in steps
// and shades a lobe around it; measuring to the step's END POINT leaves a gap
// wherever the step is longer than the lobe is wide, which on a real face drew the
// run as a chain of beads rather than a streak. Measuring to the segment makes the
// run continuous at any step count.
float seg_dist(vec2 q, vec2 a, vec2 b) {
	vec2 ab = b - a;
	float h = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
	return length(q - a - ab * h);
}

// Soft ellipse: 1 well inside, feathered to 0 at the rim.
float ell(vec2 p, vec2 c, vec2 r) {
	return 1.0 - smoothstep(0.72, 1.0, length((p - c) / max(r, vec2(1e-4))));
}

// A LOOK-HERE window: deliberately generous and very soft, so it bounds the
// search without ever drawing its own edge - whatever shape survives is the
// evidence's, not the window's.
float win(vec2 p, vec2 c, vec2 r) {
	return 1.0 - smoothstep(0.35, 1.0, length((p - c) / max(r, vec2(1e-4))));
}

// One channel's advected sample: paint drifts toward `attract` and wanders
// under the shared ooze field. Each channel advects toward ITS OWN
// attractor - the independence of the features lives here.
float advected(int ch, vec2 p, vec2 attract, vec2 asp, float dt, vec2 bias, float pull) {
	vec2 wob = vec2(vnoisep(p * 5.0 + vec2(u_time * 0.5, 3.7)),
		vnoisep(p * 5.0 + vec2(19.1, -u_time * 0.4))) - 0.5;
	vec2 v = (attract - p) * pull + wob * 0.35 + bias;
	vec2 src = clamp((p - v * dt) / asp, vec2(0.0), vec2(1.0));
	vec4 s = texture(u_prev, src);
	return ch == 0 ? s.r : (ch == 1 ? s.g : (ch == 2 ? s.b : s.a));
}

void fragment() {
	vec2 asp = vec2(u_aspect, 1.0);
	vec2 p = UV * asp;
	vec2 el = u_eye_l * asp;
	vec2 er = u_eye_r * asp;
	vec2 mo = u_mouth * asp;
	vec2 nc = u_nose * asp;
	vec2 fc = u_face_c * asp;
	// EVIDENCE, read from the frame at this texel. Everything is relative to
	// the FACE's own statistics, never absolutes - the same discipline the
	// mouth detector uses, and the only thing that survives a monochrome or
	// heavily-graded clip.
	// DILATED evidence: the max over a small ring, not the single texel.
	// Raw per-pixel evidence traced a feature's outline (the lash line, the
	// lip line) but never covered it - the paint has to fill the eye, the
	// nose, the mouth. A max-filter grows each region outward from its own
	// strongest pixels, so coverage comes from the anatomy's real extent and
	// the contour stays organic (a floor would just redraw the disc).
	float dark = 0.0;
	float bright = 0.0;
	float redness = -1.0;
	float bright0 = 0.0;   // the CENTRE tap only - see the hollow gate below
	float dark0 = 0.0;     // ...and its darkness, which is what Evidence shapes
	float red0 = 0.0;
	for (int k = 0; k < 9; k++) {
		vec2 off = vec2(0.0);
		if (k > 0) {
			float a = float(k - 1) * 0.7853981634;
			off = vec2(cos(a), sin(a)) * 0.016;
		}
		vec3 c3 = texture(u_frame, clamp(UV + off / vec2(u_aspect, 1.0),
			vec2(0.0), vec2(1.0))).rgb;
		float l3 = dot(c3, vec3(0.299, 0.587, 0.114));
		dark = max(dark, clamp((u_face_lum - l3) / max(u_face_lum, 0.08), 0.0, 1.0));
		float b3 = clamp((l3 - u_face_lum) / max(u_face_lum, 0.08), 0.0, 1.0);
		bright = max(bright, b3);
		if (k == 0) {
			bright0 = b3;
			dark0 = clamp((u_face_lum - l3) / max(u_face_lum, 0.08), 0.0, 1.0);
			red0 = c3.r - c3.g;
		}
		redness = max(redness, c3.r - c3.g);
	}
	float fred = redness;
	// EYES: the black covers what is actually DARK inside the look-window -
	// socket, lashes, lid crease, brow - so it wraps the eye's own shape
	// instead of capping a disc over it. Windows are per-eye and generous.
	// THE ANATOMY'S OWN SPACING BOUNDS EVERY FEATURE, so Scale cannot merge them.
	// Scale multiplied each radius independently and nothing knew about anything
	// else, so past a certain setting the nose window and the lip window simply
	// overlapped and the two evidence terms filled the gap between them: one huge
	// red blob where a nose and a mouth should be, and eye patches wide enough to
	// meet across the bridge and run onto the forehead. That is not Scale being
	// set too high, it is Scale having no idea the face has other features on it.
	//
	// So each radius is capped against the distance to its NEIGHBOURS - which the
	// detector already knows - at fractions that keep a real gap between windows.
	// Scale still grows everything and still reads as Scale; it now approaches the
	// packing limit of the face instead of sailing past it.
	float sep = max(distance(el, er), 1e-4);          // between the eyes
	vec2 emid = (el + er) * 0.5;
	float d_en = max(distance(emid, nc), 1e-4);       // eye line down to the nose
	float d_nm = max(distance(nc, mo), 1e-4);         // nose down to the mouth
	float spread = 1.0 + 0.7 * u_bleed;
	float rl = clamp(u_eye_lr, 0.012, 0.08) * u_scale * 1.75 * spread;
	float rr = clamp(u_eye_rr, 0.012, 0.08) * u_scale * 1.75 * spread;
	// 0.42 of the separation leaves the two eye windows clear of each other
	// (touching would need 0.5 each), and 0.72 of the eye-to-nose distance keeps
	// them off the nose.
	rl = min(rl, min(sep * 0.42, d_en * 0.72));
	rr = min(rr, min(sep * 0.42, d_en * 0.72));
	float eye_ev = smoothstep(0.04, 0.26, dark);
	float wl = win(p, el, vec2(rl, rl * 1.15));
	float wr = win(p, er, vec2(rr, rr * 1.15));
	float drip_t = 0.0;
	// The rubbed edge, shared by every feature - see the treatment inside the
	// branch. Held out here because the lips and the nose are shaded further down.
	float smudge_lip = 0.0;
	float smudge_nose = 0.0;
	float smudge_thr = 0.5;
	float smudge_band = 0.28;
	if (u_has_stencil > 0.5) {
		// THE EYE PATCH'S EDGE IS RUBBED, NOT CUT. The patch is a rasterized
		// polygon, so its edge steps from 1 to 0 across about one texel - and once
		// the interior became a flat fill (see eye_flat below) that step was the
		// whole story: "it's like you've drawn two black ovals". Nobody's eye
		// make-up ends on a line. Someone who has been crying and rubbing at it has
		// paint that thins away over a band and reaches further out in some places
		// than others, which is two separate properties and this draws both:
		//
		//   THE BAND   the polygon's alpha is averaged over a disc, which turns the
		//              one-texel step into a ramp as wide as the disc - the same
		//              trick the coat's own feather uses, and the only way to get a
		//              gradient out of an edge that has none.
		//   THE REACH  the level that ramp is cut at is modulated by noise IN THE
		//              FACE'S OWN FRAME, so the paint pushes out past the contour
		//              in some places and pulls in at others, and the pattern rides
		//              the head rather than the screen (the rule clown_anchor_check
		//              exists for).
		//
		// Both scale with Smudge. Its floor is a soft edge rather than the polygon,
		// because the polygon edge is the thing being reported.
		// BOTH EDGES ARE THE SAME EDGE. The patch's outer contour and the OPENING's
		// rim are two boundaries of one rasterized polygon pair, and a hard rim
		// around the eye reads exactly as wrong as a hard rim around the patch - so
		// one control rubs both. The opening lives in the coat's alpha (the second,
		// subtractive stencil pass cut it there), so the two channels are sampled
		// from the SAME taps: one texture fetch, two accumulators.
		float smd = clamp(u_eye_smudge, 0.0, 1.0);
		// CAPPED WELL UNDER THE OPENING'S OWN SIZE. An eye is about 0.034 tall in
		// these units, so a disc much past a third of that averages the ring's paint
		// straight across the hole and closes it.
		float sm_r = min(0.004 + 0.014 * smd, 0.016);
		vec4 c_e = texture(u_stencil, UV);
		float acc_e = c_e.r;
		// The lips and the nose come off the SAME taps - it is one texture fetch
		// either way - because they are polygons with the same hard edge and the
		// same complaint against it ("the nose looks like a hard shape instead of a
		// red smear"). One control rubs the whole make-up.
		float acc_g = c_e.g;
		float acc_b = c_e.b;
		float wsum_e = 1.0;
		for (int ei = 0; ei < 8; ei++) {
			float ea = float(ei) * 0.7853981634 + 0.3926990817;
			vec2 dir_e = vec2(cos(ea) / max(u_aspect, 1e-4), sin(ea));
			vec4 t_mid = texture(u_stencil, clamp(UV + dir_e * sm_r * 0.5,
				vec2(0.0), vec2(1.0)));
			vec4 t_out = texture(u_stencil, clamp(UV + dir_e * sm_r,
				vec2(0.0), vec2(1.0)));
			acc_e += 0.75 * t_mid.r + 0.45 * t_out.r;
			acc_g += 0.75 * t_mid.g + 0.45 * t_out.g;
			acc_b += 0.75 * t_mid.b + 0.45 * t_out.b;
			wsum_e += 1.2;
		}
		float soft_e = clamp(acc_e / wsum_e, 0.0, 1.0);
		smudge_lip = clamp(acc_g / wsum_e, 0.0, 1.0);
		smudge_nose = clamp(acc_b / wsum_e, 0.0, 1.0);
		// THE RAGGED REACH, in the FACE's own frame so it rides the head rather
		// than the screen - the rule clown_anchor_check exists for.
		vec2 fdir_e = (er - el) / max(distance(er, el), 1e-4);
		vec2 frel_e = (p - fc) / max(distance(er, el), 1e-4);
		vec2 fco_e = vec2(dot(frel_e, fdir_e), dot(frel_e, vec2(-fdir_e.y, fdir_e.x)));
		float rag_e = vnoisep(fco_e * (7.0 + 5.0 * smd) + vec2(3.1, 8.4));
		// THE RAMP IS CENTRED ON THE CONTOUR. Widening it from a fixed lower edge
		// walks the 50% point inward, so the first cut of this made the patch
		// visibly SHRINK as it softened - and "rubbed" is not "smaller". Solving
		// thr + band/2 = 0.5 keeps the average boundary exactly where the polygon
		// is at every setting; the raggedness then pushes it out past that in some
		// places and pulls it in at others, which is the other half of what a
		// worried-at edge looks like.
		float band_e = 0.28 + 0.35 * smd;
		float thr_e = 0.5 - band_e * 0.5 - (rag_e - 0.5) * 0.40 * smd;
		smudge_band = band_e;
		smudge_thr = thr_e;
		wl = smoothstep(thr_e, thr_e + band_e, soft_e);
		// SOFT PRESSURE AGAINST THE BRIDGE OF THE NOSE. The two eye patches are the
		// measured eyes grown by Eye size, and at any generous setting they grow far
		// enough toward the nose to meet over it - one bandit's mask instead of two
		// eyes. Holding the HULLS back cannot fix that without showing: sixteen
		// vertices cannot describe a soft boundary, and a growth cap, a hard flatten
		// and an asymptotic squeeze each drew a straight edge down the inner side of
		// the socket (all three tried, all three reported as a cut).
		//
		// A fade has a sample per texel, so it can be as gradual as it likes. The
		// THE NUMBERS ARE THE WHOLE OF IT, and the first pair were wrong. Full
		// strength outside 0.30 of the eye separation and zero by 0.06 puts the ramp
		// straight across the eye's own OPENING: at a wound-up Hollow the hole
		// reaches 0.13 of the separation from the midline, where that ramp is down to
		// a fifth, so the paint died along the inner edge of the hole and left the
		// hard line it was meant to remove. The fade has to be FINISHED before it
		// reaches any part of the eye. It is now full by 0.11 and zero by 0.02, so
		// it lives in the bridge itself and nowhere else - a bare strip up the middle
		// of the nose with soft shoulders, and every part of both eyes untouched.
		// Measured in EYE SEPARATIONS, so it holds on any face at any distance, and
		// along the eye line, so it holds at any tilt.
		vec2 evb = er - el;
		float sepb = max(length(evb), 1e-4);
		float alongb = abs(dot(p - (el + er) * 0.5, evb / sepb));
		wl *= smoothstep(sepb * 0.02, sepb * 0.11, alongb);
		// NO SECOND GATE ON THE COAT'S ALPHA. One was tried, to stop a wide Smudge
		// blurring paint back over the eyeball; the coat's alpha is zero in the
		// openings AND everywhere off the face, and multiplying by it cut a straight
		// edge down each socket - "a clear cut right down each eye-hole, everything
		// to the left of both eyes white". The eyeball is kept clear by the RING's
		// own geometry, which is what kept it clear before Smudge existed; all
		// Smudge has to do is not blur wider than the opening (see sm_r's cap).
		wr = 0.0;   // both eyes live in the one channel now
		// THE RUN IS A TEAR OF LINER LEAVING THE LASH LINE.
		//
		// Third pass. The first was a bundle of hairlines keyed on frame columns;
		// the second fixed the count and the taper but sized itself off `rl`, the
		// LOOK-WINDOW radius (1.75x the measured eye), and started 0.8 of that
		// BELOW the eye centre. On the author's own clip that put a lobe most of an
		// eye wide on the cheekbone with a gap between it and the eye paint: "a
		// gigantic upside-down teardrop that starts at the upper cheekbone and does
		// not blend into the eye shadow". Every number in it was a guess about a
		// face, and on a real face the guesses were wrong.
		//
		// So nothing here is guessed any more. The run is measured off THE PAINT
		// THAT IS ACTUALLY THERE - the rasterized eye patch in the stencil, which
		// already knows the eye's real outline, how far Eye size grew it and how
		// far Hollow opened it:
		//
		//   WHERE IT LEAVES  walk down-and-outward from the eye centre until the
		//                    patch ends. That is the lash line's lower-outer corner
		//                    on THIS face at THIS moment, not an offset from a
		//                    radius. The head of the run is then set slightly INSIDE
		//                    the patch, so the two are one shape and there is no
		//                    seam to blend.
		//   HOW WIDE         walk outward along the eye line until the patch ends;
		//                    the run's source width is a FRACTION of that (liner
		//                    runs as a streak a fraction of the eye's width, not as
		//                    a blob the size of it), and Drip width scales it.
		//   HOW FAR          eye to chin, scaled by Drip, as before.
		//   WHAT SHAPE       width tapers to a hairline; the path is biased outboard
		//                    by an amount that GROWS with distance, so it curves onto
		//                    the cheek rather than falling past the nose and lips;
		//                    Drip curve scales that bias; and the picture's own
		//                    structure still deflects it, so it finds the hollows.
		//
		// If the walk finds no patch at all - eyes lost to a turn, a hand, a blink
		// with Hollow wide open - there is no run. That is the gate; there is no
		// separate "is there paint" test to get wrong. (One was written, sampling
		// the eye CENTRE, which is a HOLE in a ring - it read zero everywhere and
		// Drip went completely inert.)
		//
		// It is not evidence-gated. It used to be folded into the eye window, which
		// is multiplied by how DARK the picture is - and a cheek is not dark, so the
		// run was scaled to nothing the moment it left the socket. Paint running
		// down a cheek is not evidence of anything; it is paint.
		if (u_drip > 0.001) {
			vec2 eyev_d = er - el;
			vec2 ex_d = eyev_d / max(length(eyev_d), 1e-4);   // across the face, l -> r
			// The furthest a run could sensibly go: eye to chin, measured on the face.
			float chin_d = max((u_face_c.y + u_face_r.y) - u_eye_l.y, 0.02);
			// The gradient taps' spacing. `grad` below is a luminance DIFFERENCE
			// across this, not a derivative, so the deflection's threshold is
			// calibrated to it - a real cheekbone moves luminance about 0.02 across a
			// step this size, and an earlier gate of 0.01..0.10 sat below its own
			// knee and never fired.
			vec2 gstep_d = vec2(0.0072 / max(u_aspect, 1e-4), 0.0072);
			for (int e = 0; e < 2; e++) {
				vec2 eye_c = e == 0 ? el : er;
				float eye_w = e == 0 ? rl : rr;      // the look-window: a SEARCH bound only
				float sgn = e == 0 ? -1.0 : 1.0;     // outward, along the eye line
				float reach = u_drip * chin_d;
				// Coarse rejection before the walks, on the most generous box the run
				// could occupy. Without it every texel in the frame does both walks.
				// `rl`/`rr` are CLAMPED at 0.08 height units, and a close-up face
				// measures 0.128 - so on the footage this was built against the
				// look-window is smaller than the eye it is a window onto. The box
				// takes the larger of the two, or it clips the run it is bounding.
				float ew_b = max(eye_w, e == 0 ? u_eye_lr : u_eye_rr) * 5.0;
				if (p.y < eye_c.y - ew_b || p.y > eye_c.y + ew_b + reach
						|| abs(dot(p - eye_c, ex_d)) > ew_b + reach) { continue; }
				// WALK DOWN-AND-OUT to the patch's lower-outer corner...
				float low = 0.0;
				for (int k = 1; k <= 24; k++) {
					float dd = float(k) * eye_w * 0.16;
					vec2 q = eye_c + ex_d * (sgn * dd * 0.45) + vec2(0.0, dd);
					if (texture(u_stencil, clamp(q / asp, vec2(0.0), vec2(1.0))).r > 0.40) {
						low = dd;
					}
				}
				if (low < 1e-5) { continue; }        // no eye paint here, so no run
				// ...and OUTWARD to its outer edge, which is what sets the width.
				float halfw = 0.0;
				for (int k = 1; k <= 14; k++) {
					float dd = float(k) * eye_w * 0.28;
					vec2 q = eye_c + ex_d * (sgn * dd);
					if (texture(u_stencil, clamp(q / asp, vec2(0.0), vec2(1.0))).r > 0.40) {
						halfw = dd;
					}
				}
				if (halfw < 1e-5) { halfw = low; }
				// THE RUN LEAVES AT THE LASH LINE'S OWN WIDTH. An earlier cut made it
				// 0.16 of that at the top of its slider, on the theory that a tear is
				// a fraction of an eye's width - true of a tear, wrong here, because
				// what has to read is the JOIN: paint leaving a socket leaves along
				// the whole edge it is draining and only then gathers into a run. A
				// sixth of the socket's width reads as a thread stuck underneath it.
				// `u_drip_w` is a direct fraction of the measured half-width now, and
				// at the top of its slider the run leaves wider than the socket's own
				// half-width.
				float w0 = halfw * clamp(u_drip_w, 0.05, 1.6);
				// The head sits INSIDE the patch - that overlap is what makes the run
				// and the eye shadow read as one shape instead of two.
				// ...capped at half the way back to the eye centre, since w0 is now a
				// socket-sized number and burying the head by all of it would start
				// the run above the eye.
				// ...capped at half the way back to the eye centre, since w0 is a
				// socket-sized number and burying the head by all of it would start
				// the run above the eye.
				vec2 srcp = eye_c + ex_d * (sgn * low * 0.45)
					+ vec2(0.0, low - min(w0 * 0.9, low * 0.5));
				float seg = reach / 18.0;
				vec2 cp = srcp;
				float run = 0.0;
				for (int d = 0; d < 18; d++) {
					float t = (float(d) + 0.5) / 18.0;
					vec2 gp = clamp(cp / asp, vec2(0.0), vec2(1.0));
					float l_c = dot(texture(u_frame, gp).rgb, vec3(0.299, 0.587, 0.114));
					float l_x = dot(texture(u_frame, clamp(gp + vec2(gstep_d.x, 0.0),
						vec2(0.0), vec2(1.0))).rgb, vec3(0.299, 0.587, 0.114));
					float l_y = dot(texture(u_frame, clamp(gp + vec2(0.0, gstep_d.y),
						vec2(0.0), vec2(1.0))).rgb, vec3(0.299, 0.587, 0.114));
					vec2 grad = vec2(l_x - l_c, l_y - l_c);
					vec2 tang = vec2(-grad.y, grad.x);
					if (tang.y < 0.0) { tang = -tang; }   // a run only ever goes downhill
					float gs = smoothstep(0.002, 0.020, length(grad));
					vec2 dir = normalize(vec2(0.0, 1.0)
						+ ex_d * (sgn * clamp(u_drip_curve, 0.0, 2.0) * t)
						+ normalize(tang + vec2(0.0, 1e-3)) * (0.22 * gs));
					vec2 pcp = cp;
					// STEP BY VERTICAL DESCENT, not by arc length. Drip promises "this
					// far down the face"; dividing the step by the direction's own
					// vertical component keeps that promise whatever Drip curve is set
					// to, instead of a curved run quietly falling short of a straight
					// one. Floored so a step that has turned nearly sideways cannot
					// run away.
					cp += dir * (seg / max(dir.y, 0.35));
					// THE TAPER, plus a ragged term so it is not a drawn wedge. Both
					// are functions of position ALONG THE RUN, so they ride the head.
					float ragg = vnoisep(vec2(t * 9.0 + float(e) * 31.0, 2.7));
					// FULL WIDTH AT THE TOP, THEN A FAST THIN. A linear taper spends
					// most of the run at a middling width, which is what read as "thin
					// the entire length": no stretch of it was actually joined to the
					// eye at the eye's width. A smoothstep holds the source width long
					// enough for the join to read, then collapses over the first half.
					// THE FLOOR IS NOT A HAIRLINE, though - a first cut collapsed to
					// 6% by 42% of the run, and clown_controls_check immediately caught
					// what that costs: Drip curve went dead, because a control that
					// moves the run sideways has nothing to move once five-sixths of
					// the run is a sub-pixel thread. The tail stays a visible line.
					float w = w0 * (0.14 + 0.86 * (1.0 - smoothstep(0.05, 0.55, t)))
						* (0.80 + 0.40 * ragg);
					// A DARK THREAD INSIDE A SOFT SMUDGE, not a flat bar. A single
					// hard-edged lobe is what made the run read as a stripe drawn
					// with a marker: pigment carried in liquid does not have one
					// width, it has a wet core and a halo where it has spread and
					// thinned. Two profiles off the same centreline give that for
					// one distance calculation.
					float dd = seg_dist(p, pcp, cp);
					float core = 1.0 - smoothstep(w * 0.10, w * 0.45, dd);
					float halo = 1.0 - smoothstep(w * 0.30, w * 1.30, dd);
					float lobe = max(core, halo * 0.45);
					// Thinning in WIDTH is what reads as a run drying out; the value
					// eases off over the last third as well, so the tail fades into
					// the cheek instead of stopping square.
					run = max(run, lobe * smoothstep(1.0, 0.78, t) * mix(1.0, 0.70, t));
				}
				// THE EYEBALL STAYS CLEAR. The run leaves at the socket's own width
				// now, so its head reaches back UP into the ring's opening - over the
				// eyeball, the one place the eye paint deliberately does not go, and
				// the boundary the rest of the mask respects. Above the drain line the
				// run is therefore admitted ONLY where the stencil says there is eye
				// paint (`wl` is exactly that, already sampled for the socket); below
				// it, it is free.
				//
				// Masking against the STENCIL rather than against a reconstructed hole
				// is what makes this exact. The obvious alternative - rebuild the hole
				// from (0.9 + Hollow * 0.6) x the measured radius, which the shader has
				// - assumes the opening is a disc. It is not: the measured radius is
				// the eye's SEMI-MAJOR extent and an eye is about 0.4 as tall as it is
				// wide, so a disc that clears the eyeball also eats the whole head of
				// the run. The stencil already holds the real outline, at whatever Eye
				// size and Hollow have made it, on a head at whatever angle.
				float above_src = smoothstep(srcp.y + 0.004, srcp.y - 0.004, p.y);
				drip_t = max(drip_t, run * max(1.0 - above_src, wl));
			}
		}
	}
	// HOLLOW's radial hole, gated on brightness: a visible eyeball is the bright
	// thing inside a dark socket, so the hole opens on an open eye and mostly
	// closes on a blink - which is what stops the paint flickering shut every time
	// she blinks. The gate reads THIS pixel's own brightness, never the dilated
	// max: a closed eyelid sits next to bright cheek, and a dilated read opened
	// the hole and erased the eye paint on every blink.
	float hole_gate = smoothstep(0.05, 0.30, bright0);
	float dnl = length((p - el) / max(vec2(rl, rl * 1.15), vec2(1e-4)));
	float dnr = length((p - er) / max(vec2(rr, rr * 1.15), vec2(1e-4)));
	float hole_e = smoothstep(0.55, 0.0, min(dnl, dnr)) * hole_gate;
	// EVIDENCE, REBUILT. It used to be a FLOOR under the evidence term - a fixed
	// share of each feature painted whatever the picture said - and measured over
	// the whole pipeline it moved the output by 0.0004, which is nothing. The
	// reason is that the contours made it redundant: a floor only matters where
	// the evidence is WEAK, and back when the windows were ellipses they spilled
	// onto plain skin where it was. Now the window IS the feature, so the evidence
	// is strong nearly everywhere inside it and the floor never gets a say.
	//
	// So it now shapes the evidence instead of flooring it, which is a question
	// the contours did NOT already answer: how much of the picture's own light and
	// shade shows through the paint. Low is a flat fill of the measured outline;
	// high paints only where the frame is genuinely dark or red, so the eye's
	// crease and the lip line come through as texture.
	// It shapes the CENTRE tap, not the dilated one. `dark`, `bright` and
	// `redness` are each a MAX over a ring of nine taps - a dilation, there so the
	// paint covers a feature robustly rather than pinholing on it - and a dilated
	// max saturates: measured, every evidence term sat at 1.0 across the whole
	// window, so shaping them changed the output by 0.0007, which is nothing. The
	// centre tap is the one that still VARIES pixel to pixel, so Evidence blends
	// from the dilated term (a flat fill of the outline) toward the centre tap's
	// own value (the picture's texture showing through).
	float ev_mix = clamp(u_evidence, 0.0, 1.0);
	// The socket's paint is evidence-shaped; the run below it is not (see the drip
	// above). Hollow's radial hole is scaled back when the stencil is present -
	// the eyeball is cut out of the RING geometrically there, and applying both
	// would open the hole twice.
	float hole_k = u_has_stencil > 0.5 ? 0.35 : 1.0;
	// THE FLOOR STAYS on the flat end. Dropping it entirely - which the first cut
	// of this did - makes a feature vanish wherever the evidence is weak, and
	// clown_scale_check caught exactly that: the nose fell to 0.14 at low Scale.
	// The floor is what guarantees a feature is PAINTED at all. So Evidence blends
	// between two live behaviours - a reliable fill at 0 (the original look) and
	// the picture's own texture at 1 - rather than between a fill and nothing.
	float eye_tex = smoothstep(0.02, 0.42, dark0);
	// THE FLAT END IS ACTUALLY FLAT, once there is a measured outline. Evidence's
	// own doc above promises "a flat fill of the measured outline" at 0 - and this
	// was 0.22 + 0.78 * eye_ev, which is not a fill, it is the evidence again with
	// a floor under it. So on the bright skin OUTSIDE an eye - the temple, the
	// outer corner - the patch fell to 0.22, and mask_split's eyeM16 ramp
	// (smoothstep(0.12, 0.50, r)) turns 0.22 into 26% black: the white coat came
	// through the middle of the eye paint and split it into two halves, which is
	// exactly what was reported. It painted the lashes and the socket crease and
	// missed everything between them.
	//
	// With the stencil the window IS the feature - a measured contour of that eye,
	// grown by Eye size - so filling it is the whole point, and Evidence is what
	// lets the picture back through. WITHOUT one, `wl` is a deliberately generous
	// analytic ellipse and a flat fill of THAT is the "circles drawn over the face"
	// complaint, so the old evidence-shaped behaviour is what the fallback keeps.
	float eye_flat = u_has_stencil > 0.5 ? 1.0 : (0.22 + 0.78 * eye_ev);
	float tblack = max(wl, wr) * mix(eye_flat, eye_tex * eye_tex, ev_mix)
		* (1.0 - u_hollow * hole_e * hole_k);
	// FULL VALUE. The run already carries its own falloff (see the deposit), and
	// scaling it again here is what put a visible step between the socket and the
	// paint leaving it - the two are one shape, so they are one value.
	tblack = max(tblack, drip_t);
	// LIPS: redder than the face's own mean, or the dark line of the mouth
	// itself - whichever is stronger. The look-window is wide and shallow
	// (a mouth's proportions), the evidence draws the lips.
	// Capped in HEIGHT units, same reason as the coat's radii below - a raw-uv
	// cap on a width is a different physical width on every frame shape, and it
	// squeezed the lips to a third of their span on portrait footage.
	vec2 lr = vec2(clamp(u_mouth_r.x * u_aspect, MOUTH_CAP_XLO, MOUTH_CAP_XHI) * 1.9,
		clamp(u_mouth_r.y, 0.010, 0.032) * 1.7) * u_scale * spread;
	// The lips' HEIGHT is what runs into the nose - their width has nothing above
	// it to collide with, so it is bounded only against the eye separation to stop
	// a mouth wider than the face.
	lr = vec2(min(lr.x, sep * 1.05), min(lr.y, d_nm * 0.5));
	float lip_ev = max(smoothstep(0.005, 0.075, fred - u_face_red),
		smoothstep(0.07, 0.38, dark) * 0.9);
	// Same hollow rule for the lips: teeth are the bright thing inside an
	// open mouth, so the centre clears when it opens and fills when shut.
	float hole_m = smoothstep(0.45, 0.0, length((p - mo) / max(lr, vec2(1e-4)))) * hole_gate;
	float lipw = win(p, mo, lr);
	if (u_has_stencil > 0.5) {
		// The measured lips, already widened and swept by Smile width / Smile
		// curve before rasterizing - so a Joker mouth is a real mouth's outline
		// stretched, not an ellipse pretending.
		lipw = smoothstep(smudge_thr, smudge_thr + smudge_band, smudge_lip);
	}
	float lip_tex = max(smoothstep(0.0, 0.10, red0 - u_face_red),
		smoothstep(0.04, 0.45, dark0) * 0.85);
	float lip_flat = 0.18 + 0.82 * lip_ev;
	float tlip = lipw * mix(lip_flat, lip_tex * lip_tex, ev_mix) * (1.0 - u_hollow * hole_m);
	// NOSE: the one feature with no reliable 2D signature - a nose is a 3D
	// protrusion, so its evidence is the LIT RIDGE plus the shadow under the
	// tip. Window-led with a higher floor, brightness-shaped so the ball
	// still hugs the actual nose rather than sitting as a decal.
	float nr = clamp(rl * 0.55 + rr * 0.55, 0.018, 0.075) * u_scale;
	// The nose sits BETWEEN the eye line and the mouth and is the feature with a
	// neighbour on both sides, so it takes the tightest bound of the three - half
	// the way down to the lips, and clear of the eye windows above it.
	nr = min(nr, min(d_nm * 0.48, d_en * 0.55));
	// Brightness leads. The shadow UNDER the nose is dark evidence too, and
	// leaning on it bridged the ball into the lips - so the dark term is a
	// weak, late-thresholded assist only, and the window stays compact.
	float nose_ev = max(smoothstep(0.0, 0.22, bright), smoothstep(0.22, 0.55, dark) * 0.45);
	float nosew = win(p, nc, vec2(nr, nr * 0.95));
	if (u_has_stencil > 0.5) {
		nosew = smoothstep(smudge_thr, smudge_thr + smudge_band, smudge_nose);
	}
	float nose_tex = max(smoothstep(0.0, 0.30, bright0), smoothstep(0.25, 0.6, dark0) * 0.5);
	float nose_flat = 0.42 + 0.58 * nose_ev;
	float tnose = nosew * mix(nose_flat, nose_tex * nose_tex, ev_mix);
	// The coat's radii are CAPPED at a face-plausible size: on footage where
	// the key hue washes the whole room, the fitted spread is the ROOM's,
	// and an uncapped coat painted the wallpaper (again - the same lesson as
	// the old anchor-scale cap).
	//
	// THE CAP IS APPLIED IN HEIGHT UNITS, after the aspect conversion, never in
	// raw uv before it. A bound written in raw uv is a different PHYSICAL size on
	// every frame shape: 0.145 of the width is 0.258 frame-heights on 16:9 and
	// 0.082 on a 9:16 phone clip, so the same cap that comfortably clears a face
	// on one strangled it to a third of its width on the other - a narrow
	// vertical sliver about a fifth the area of the real face, which is exactly
	// what portrait footage produced. FACE_CAP_* are the old raw bounds converted
	// at 16:9 (x1.7778), so landscape clips fit identically to before and every
    // other shape now gets the same physical cap instead of a scaled one.
	vec2 fr = vec2(clamp(u_face_r.x * u_aspect, FACE_CAP_XLO, FACE_CAP_XHI),
		clamp(u_face_r.y, 0.07, 0.205))
		// Scale's authority over the COAT, as opposed to over the features. It
		// was 0.75 + 0.25x, a 1.7:1 span across the whole slider - and with both
		// radii sitting on their caps (which they do on any clip where the fit
		// is generous) that reads as a control that does nothing. Same value at
		// the default 1.0, more travel either side of it.
		* 1.05 * (0.4 + 0.6 * u_scale);
	float tface = 1.0 - smoothstep(0.72, 1.05, length((p - fc) / fr));
	if (u_has_stencil > 0.5) {
		// The jawline itself, so the coat stops at the face instead of at an
		// ellipse that is too wide at the temples and too narrow at the chin -
		// and FEATHERED, by averaging the silhouette over a small disc. Two rings
		// plus the centre: enough taps that the ramp is smooth rather than
		// banded, few enough to be free. Round in pixels (u_texel_c carries the
		// aspect), so the feather is the same width top and side.
		float acc_f = texture(u_stencil, UV).a;
		float wsum_f = 1.0;
		for (int fi = 0; fi < 8; fi++) {
			float fa = float(fi) * 0.7853981634 + 0.3926990817;
			vec2 fd = vec2(cos(fa) / max(u_aspect, 1e-4), sin(fa)) * u_coat_feather;
			acc_f += 0.75 * texture(u_stencil, clamp(UV + fd * 0.5, vec2(0.0), vec2(1.0))).a;
			acc_f += 0.45 * texture(u_stencil, clamp(UV + fd, vec2(0.0), vec2(1.0))).a;
			wsum_f += 1.2;
		}
		tface = clamp(acc_f / wsum_f, 0.0, 1.0);
	}
	vec4 tgt = vec4(tblack, tlip, tnose, tface);
	if (u_reset == 1) {
		COLOR = tgt;
	} else {
		// Carry: advection toward each channel's OWN attractor (the eye
		// channel flows to whichever eye is nearer, the coat to the face
		// centre), folded with a touch of blur so edges soften in transit.
		vec2 eatt = distance(p, el) < distance(p, er) ? el : er;
		// THE EYE PAINT RUNS DOWNWARD. Every other channel just flows to its own
		// attractor, which is isotropic - paint that ends up above an eye creeps
		// back down at exactly the rate paint below it creeps up, so a patch that
		// overshoots reads as a smear across the brow and forehead. A clown's eye
		// paint has a direction: it sits on the eye and runs down the cheek. A
		// small constant downward term in the eye channel's drift is the whole
		// of it - equilibrium settles just below the socket, and anything that
		// strays upward is pulled back down instead of lingering.
		// THE COAT BARELY DRIFTS, and the features still do. Advection samples
		// BACKWARD along the flow, so a channel pulled toward the face centre
		// reads, at its own outer boundary, from just OUTSIDE itself - where
		// there is no paint. The deposit refills that from zero every step, so
		// the rim sits at whatever one step of refill reaches and never catches
		// up: a permanent unpainted band around the edge of the coat, as deep as
		// the refill is slow. Measured against the jawline it was 3.7% of frame
		// width at the default Settle and 13% at the top of the slider, which is
		// exactly the shape of the report ("settle makes that region wider").
		//
		// The features NEED that drift - it is what makes orphaned paint flow to
		// a re-fit rather than fade in place. The coat does not: it is a FILL, it
		// covers whatever the face oval currently is, and the deposit already
		// follows the face every frame. So the pull is per-channel now, and the
		// coat's is small enough that its boundary is fed rather than drained.
		vec4 adv = vec4(
			advected(0, p, eatt, asp, u_dt, vec2(0.0, EYE_DRIP), 2.4),
			advected(1, p, mo, asp, u_dt, vec2(0.0), 2.4),
			advected(2, p, nc, asp, u_dt, vec2(0.0), 2.4),
			advected(3, p, fc, asp, u_dt, vec2(0.0), COAT_PULL));
		vec2 px = 1.0 / vec2(textureSize(u_prev, 0));
		vec4 blur = (texture(u_prev, UV + vec2(px.x, 0.0))
			+ texture(u_prev, UV - vec2(px.x, 0.0))
			+ texture(u_prev, UV + vec2(0.0, px.y))
			+ texture(u_prev, UV - vec2(0.0, px.y))) * 0.25;
		vec4 carried = mix(adv, blur, mix(0.10, 0.45, u_bleed));
		// SETTLE sets both time constants: how long orphaned paint lingers,
		// and how fast new paint rises. The deposit is RATE-LIMITED, not
		// instant - slamming to full value the moment evidence spiked (and
		// then decaying slowly) is exactly what made the feature shapes
		// pulse frame to frame. Rising over ~0.1-0.4s low-passes the
		// evidence's own noise without visibly lagging the face.
		float tau_dec = mix(0.25, 1.6, clamp(u_settle, 0.0, 1.0));
		float tau_up = mix(0.06, 0.40, clamp(u_settle, 0.0, 1.0));
		vec4 f = carried * exp(-u_dt / tau_dec);
		vec4 rise = max(vec4(0.0), tgt - f) * (1.0 - exp(-u_dt / tau_up));
		vec4 out4 = max(f + rise, vec4(0.0));
		// THE COAT NEVER FALLS BELOW ITS OWN DEPOSIT. Weakening its drift (see
		// COAT_PULL) halved the unpainted rim but could not remove it, because
		// ANY transport at a boundary with nothing beyond it drains that
		// boundary: the wander term still pushes outward sometimes, and the blur
		// still averages in the zeros outside. Whatever is lost gets refilled at
		// a rate Settle sets, so a rim always survives and always widens with
		// Settle - which is the whole of the report.
		//
		// The fix is to stop treating the coat as a liquid at its edge. The
		// features ARE paint and must be erodable (Hollow opens them, evidence
		// shapes them); the coat is a MASK - a statement of where the face is -
		// and a face does not have a permanent unpainted ring around it because
		// the paint drifted inward. So the alpha channel takes the deposit as a
		// FLOOR - and, since the openings, a CEILING too. Letting the sim add above
		// the deposit was where Bleed's outward coat spread lived, but it is also
		// what would refill the eye openings the moment they were cut out of the
		// silhouette: advection reads from just outside a hole, and just outside a
		// hole the coat is solid, so a few steps of it put a soft grey blob back
		// over the eyeball. The coat is a MASK - the sentence above already says so
		// - so it is the deposit, exactly, and every question about where it lies
		// is answered by the polygon. Bleed keeps its whole effect on the features
		// (see `spread`, which widens every feature radius).
		out4.a = tgt.a;
		COLOR = out4;
	}
}
