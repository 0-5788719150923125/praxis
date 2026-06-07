/**
 * Praxis Web - wheel-driven momentum scrolling.
 *
 * Mobile touch already coasts natively; desktop wheels step. This converts
 * wheel input on a scrollable into the same friction coast the chart deck uses
 * for touch flicks (shared constants below), so every input modality has one
 * feel. Gated on INPUT TYPE (wheel events), never on platform - a wheel
 * plugged into a phone coasts identically.
 */

// Shared scroll-coast feel: exponential friction halflife + settle threshold.
// charts.js imports these for the deck's touch coast - one source of truth.
export const SCROLL_TAU = 140;      // ms
export const SCROLL_MIN_VEL = 0.02; // px/ms

// Wheel feel: notches push a TARGET position (travel = delta * WHEEL_GAIN, so
// the glide always matches the scrolling effort); velocity ramps toward the
// gap-closing rate through its own ease instead of jumping per notch. Ramp-in
// over ~TAU_RAMP, cruise/decay over ~TAU_GLIDE: spin hard -> big gap -> fast,
// long glide; one notch -> a short, gentle ease.
const WHEEL_GAIN = 1.5;   // glide distance per px of wheel input
const TAU_RAMP = 90;      // ms; velocity ramp-in (the "slow start")
const TAU_GLIDE = 240;    // ms; gap-closing horizon (the decay tail)

// deltaMode normalization: Firefox reports wheel deltas in LINES (mode 1,
// deltaY ~ 3/notch) or PAGES (mode 2), not pixels. Untranslated, a line-mode
// notch became a ~3px impulse - i.e. no visible scrolling at all.
const LINE_PX = 40;             // ~3 lines/notch * 40px = a native-sized step

function wheelDeltaPx(e, el) {
    if (e.deltaMode === 1) return e.deltaY * LINE_PX;
    if (e.deltaMode === 2) return e.deltaY * el.clientHeight * 0.9;
    return e.deltaY;
}

/**
 * Attach a momentum coast to a scrollable's wheel input. Idempotent - safe to
 * call on every render; rebuilding the element's CHILDREN keeps the handler,
 * a recreated element gets a fresh one.
 */
export function attachWheelMomentum(el) {
    if (!el || el._wheelMomentum) return;
    el._wheelMomentum = true;
    let vel = 0, pos = 0, target = 0, raf = 0, last = 0;

    const tick = (t) => {
        if (!last) last = t;
        let dt = t - last; last = t;
        if (dt > 50) dt = 50;
        const max = el.scrollHeight - el.clientHeight;
        if (target < 0) target = 0;             // glide never aims past an edge
        else if (target > max) target = max;
        // Second-order ease: velocity RAMPS toward the gap-closing rate (no
        // per-notch jump), then position integrates it. Both stages use the
        // closed-form exponential, so the curve is framerate-independent.
        const want = (target - pos) / TAU_GLIDE;   // px/ms that closes the gap in ~tau
        vel += (want - vel) * (1 - Math.exp(-dt / TAU_RAMP));
        pos += vel * dt;
        // Hard clamp at the edges - but only kill velocity pointed INTO the
        // edge (resting at the top must not zero a downward ramp-in).
        if (pos <= 0) { pos = 0; if (vel < 0) vel = 0; }
        else if (pos >= max) { pos = max; if (vel > 0) vel = 0; }
        el.scrollTop = pos;
        const settled = Math.abs(target - pos) < 0.5 && Math.abs(vel) < SCROLL_MIN_VEL;
        if (settled) { pos = target; el.scrollTop = pos; }
        raf = settled ? 0 : requestAnimationFrame(tick);
    };

    el.addEventListener('wheel', (e) => {
        if (e.ctrlKey) return;                  // pinch-zoom gesture - stay native
        if (el.scrollHeight - el.clientHeight <= 0) return;  // nothing to scroll
        e.preventDefault();
        if (!raf) {                             // resync after native/keyboard moves
            pos = el.scrollTop;
            target = pos;
            vel = 0;
        }
        target += wheelDeltaPx(e, el) * WHEEL_GAIN;
        if (!raf) { last = 0; raf = requestAnimationFrame(tick); }
    }, { passive: false });

    // A real touch owns the element natively; kill any in-flight glide so the
    // two never fight over scrollTop.
    el.addEventListener('touchstart', () => { vel = 0; target = pos; }, { passive: true });
}
