/**
 * Praxis Web - Tab slide transition (mobile)
 *
 * A horizontal slide between tabs on swipe. The trick for the heavy chart tabs
 * (Research/Dynamics): the incoming panel is parked fully off-screen and left to
 * SETTLE there (Chart.js canvases re-fit on show, decks re-measure or even
 * rebuild) while the outgoing panel stays put as a cover - then it slides in
 * already painted. Letting that settle happen on-screen is what caused the
 * mid-slide stall and the appears/disappears flicker. Inline styles only, fully
 * torn down on completion; desktop / reduced-motion keep the instant swap.
 */

const SLIDE_MS = 280;
const EASE = `transform ${SLIDE_MS}ms cubic-bezier(0.22, 0.61, 0.36, 1)`;
// Whitespace held between the two panels at the seam (twice the app-container's
// horizontal padding - the two tabs' notional margins meeting), so their content
// never touches edge-to-edge mid-slide. The incoming still lands at exactly 0.
const GUTTER = '2.75rem';

// The slide in flight. A fast second swipe finalizes it before starting the
// next, so there's never an orphaned overlay or stacked transition.
let active = null;

/** Mobile + motion allowed; otherwise the caller keeps the instant swap. */
export function canSlideTabs() {
    return window.innerWidth <= 768
        && window.matchMedia('(prefers-reduced-motion: no-preference)').matches;
}

// Freeze the prism logo's render loop for the duration of a slide - one less
// thing repainting while the transition composites.
function pausePrism(on) { window.__prismPaused = on; }

function finalize(slide) {
    if (!slide || slide.done) return;
    slide.done = true;
    clearTimeout(slide.timer);
    if (slide.onEnd) slide.incoming.removeEventListener('transitionend', slide.onEnd);
    slide.outgoing.removeAttribute('style');
    slide.incoming.style.transition = '';
    slide.incoming.style.transform = '';
    slide.incoming.style.willChange = '';
    if (active === slide) active = null;
    pausePrism(false);
    // A ready tab slid on its cached content; revalidate now, underneath.
    if (slide.after) slide.after();
}

// Finish any slide still in flight, returning its panels to rest. The caller
// runs this before capturing geometry for a new switch, so a fast double-swipe
// never measures a panel that's currently parked off-screen.
export function finishActiveSlide() { if (active) finalize(active); }

/**
 * Slide `incoming` in and `outgoing` out. `dir > 0` enters from the right
 * (forward), `< 0` from the left (back). `outRect` is the outgoing panel's box,
 * captured before render() hid it. `prepare` (optional, may be async) is the
 * caller's post-switch work. `ready` says the incoming is already rendered: when
 * true we slide its cached content immediately and run `prepare` AFTER, as a
 * background revalidate - no forced wait. When false (a cold tab) `prepare` runs
 * under the cover first, so the tab is built before it slides in.
 */
export function slideTabs(outgoing, outRect, incoming, dir, prepare, ready) {
    finishActiveSlide();  // a prior slide still settling: finish it clean
    const run = () => Promise.resolve(prepare && prepare());
    if (!outgoing || !incoming || !outRect) { run(); return; }

    const sign = dir < 0 ? -1 : 1;
    // Off-screen rest positions, offset by the gutter so the panels keep a gap
    // at the seam. `enter` is where the incoming waits and slides from; `exit`
    // is where the outgoing slides to.
    const enter = sign > 0 ? `calc(100% + ${GUTTER})` : `calc(-100% - ${GUTTER})`;
    const exit = sign > 0 ? `calc(-100% - ${GUTTER})` : `calc(100% + ${GUTTER})`;

    // Cover: pin the outgoing panel as a fixed overlay where it sat (escapes the
    // app clip). render() dropped its .active class, so restore that rule's
    // column layout inline - otherwise it falls back to flex-direction:row and
    // its children squash into tall skinny strips. This keeps the old tab on
    // screen while the new one settles.
    Object.assign(outgoing.style, {
        position: 'fixed', margin: '0',
        left: `${outRect.left}px`, top: `${outRect.top}px`,
        width: `${outRect.width}px`, height: `${outRect.height}px`,
        boxSizing: 'border-box', display: 'flex', flexDirection: 'column',
        overflowY: 'auto', overflowX: 'hidden', zIndex: '5', pointerEvents: 'none',
        willChange: 'transform', transform: 'translateX(0)',
    });

    // Park the incoming fully off-screen for the whole settle phase: its chart
    // re-fit / deck rebuild then happens unseen instead of flashing in place. It
    // still measures correctly there - translateX shifts x only, not size.
    incoming.style.willChange = 'transform';
    incoming.style.transition = 'none';
    incoming.style.transform = `translateX(${enter})`;
    void incoming.offsetWidth;  // commit the off-screen start before settling

    const slide = { outgoing, incoming, done: false, timer: 0, onEnd: null, after: null };
    active = slide;
    pausePrism(true);

    // Motion: both ease across, flush at the seam - outgoing out one way,
    // incoming in behind it. The start transforms are already committed (frames
    // passed during settle), so setting transition + end transform animates.
    const startMotion = () => {
        if (slide.done) return;
        outgoing.style.transition = EASE;
        incoming.style.transition = EASE;
        outgoing.style.transform = `translateX(${exit})`;
        incoming.style.transform = 'translateX(0)';
        slide.onEnd = (e) => {
            if (e.target === incoming && e.propertyName === 'transform') finalize(slide);
        };
        incoming.addEventListener('transitionend', slide.onEnd);
        slide.timer = setTimeout(() => finalize(slide), SLIDE_MS + 80);
    };

    // Ready tab: don't wait on data - slide its cached content after a two-frame
    // settle (lets the show-triggered chart re-fit flush off-screen), and run the
    // refresh afterward. Cold tab: build it under the cover first, then slide.
    const settleThenMove = () => {
        if (slide.done) return;
        requestAnimationFrame(() => requestAnimationFrame(startMotion));
    };
    if (ready) {
        slide.after = () => { run(); };
        settleThenMove();
    } else {
        run().then(settleThenMove);
    }
}
