/**
 * Praxis Web - Mobile Optimizations
 * Clean tab carousel - clicked tabs float to left
 */

import { state } from './state.js';
import { executeAction } from './actions.js';

// renderTabs lays the tab set out this many times on mobile. Native momentum
// scroll is smooth but can't reposition mid-fling on every platform, so we keep
// generous buffer copies (a fling can't reach an end) and re-center toward the
// middle copy whenever the scroll settles into an outer copy. Odd so there's a
// true middle copy with equal slack on each side.
export const TAB_LOOP_COPIES = 5;

function tabStrip() {
    const el = document.querySelector('.tab-buttons');
    return el && window.innerWidth <= 768 ? el : null;
}

// Width of a single copy of the tab set (period of the loop), measured from the
// buttons. 0 if the strip isn't laid out as the expected multiple yet.
function copyWidth(strip) {
    const btns = strip.querySelectorAll('.tab-button');
    const n = btns.length / TAB_LOOP_COPIES;
    if (!Number.isInteger(n) || n < 1) return 0;
    return btns[n].offsetLeft - btns[0].offsetLeft;
}

/**
 * Seat the middle copy's active tab at the strip's left edge, so a full copy of
 * the set sits to either side and the carousel can scroll both ways at once.
 * Called by renderTabs after every rebuild (the strip is rebuilt on switch).
 */
export function centerLoopedTabs() {
    const strip = tabStrip();
    if (!strip) return;
    const btns = strip.querySelectorAll('.tab-button');
    const n = btns.length / TAB_LOOP_COPIES;
    if (!Number.isInteger(n) || n < 1) return;
    const mid = Math.floor(TAB_LOOP_COPIES / 2) * n;   // first button of the middle copy
    let target = btns[mid];
    for (let i = mid; i < mid + n; i++) {
        if (btns[i].classList.contains('active')) { target = btns[i]; break; }
    }
    // Relative measure (getBoundingClientRect) so it's correct regardless of the
    // button's offsetParent; aligns the active tab flush with the strip's left.
    strip.scrollLeft += target.getBoundingClientRect().left - strip.getBoundingClientRect().left;
}

// Back-compat alias: the swipe-to-switch handler re-centers after a switch.
export const snapActiveTabIntoView = centerLoopedTabs;

/**
 * Setup the mobile tab carousel as an infinite loop on top of native scroll.
 * renderTabs lays the tab set out TAB_LOOP_COPIES times; the browser handles the
 * drag and momentum (smooth, no pop-in), and we just nudge the scroll back by one
 * copy-width whenever it settles into an outer copy. The content is identical a
 * copy-width over, so the nudge is invisible - the strip loops endlessly. The wide
 * buffer means a fling can't reach an end before the next scroll tick re-centers.
 */
export function setupTabCarousel() {
    const strip = document.querySelector('.tab-buttons');
    if (!strip || window.innerWidth > 768) return;

    // Right edge fade only: it hints "more tabs ->" without overlaying the
    // leading (left-most) active tab, which must always be fully visible.
    const tabNav = document.querySelector('.tab-nav');
    if (tabNav) {
        tabNav.classList.add('has-scroll-right');
        tabNav.classList.remove('has-scroll-left');
    }

    // Bound once: #tab-buttons persists across renderTabs rebuilds (only its
    // children change). Keep the scroll parked in the central copies - jump by a
    // whole copy-width (identical pixels) when it drifts into the outer ones.
    strip.addEventListener('scroll', () => {
        const w = copyWidth(strip);
        if (!w) return;
        const sl = strip.scrollLeft;
        if (sl < 1.5 * w) strip.scrollLeft = sl + w;
        else if (sl > 3.5 * w) strip.scrollLeft = sl - w;
    }, { passive: true });

    centerLoopedTabs();
    // Re-center once fonts settle: their late metrics widen the buttons, which
    // would otherwise leave the leading tab measured at a stale offset and clipped.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(centerLoopedTabs);

    // Re-center only on real width changes - the URL bar showing/hiding during a
    // scroll fires resize and would otherwise jerk the strip mid-drag.
    let lastWidth = window.innerWidth;
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768 || window.innerWidth === lastWidth) return;
        lastWidth = window.innerWidth;
        centerLoopedTabs();
    });
}

/**
 * Switch tabs by horizontal swipe over the content area (mobile only).
 * Swipe left -> next tab, swipe right -> previous tab. Order follows
 * state.tabs, so it matches the tab strip exactly.
 */
export function setupTabSwipe() {
    // A swipe must be decisively horizontal and long enough to beat a scroll or
    // a tap. Vertical-dominant gestures fall through to native scrolling.
    const MIN_DISTANCE = 60;   // px of horizontal travel
    const MAX_OFF_AXIS = 0.6;  // |dy| must stay under 60% of |dx|

    let start = null;

    // Capture-phase on the document: this runs BEFORE any inner card/deck
    // handler, so an element that scrolls (or stops propagation) can't swallow
    // the gesture. We only read coordinates and never preventDefault, so inner
    // scrolling still works normally - we just also measure the net swipe.
    document.addEventListener('touchstart', (e) => {
        if (window.innerWidth > 768 || e.touches.length !== 1) { start = null; return; }
        // Ignore only swipes beginning on the tab strip (it owns its own
        // horizontal scroll). Everything else - including inner-scrolling cards
        // and the vertical chart deck - is fair game: a horizontal swipe is
        // orthogonal to their vertical gestures, so the off-axis filter on
        // release keeps them from conflicting.
        // Only swipe when the tab strip is on screen and the touch lands in the
        // app body - not on the strip itself (owns its scroll) or over a modal.
        if (e.target.closest('.tab-nav')) { start = null; return; }
        if (!e.target.closest('.app-container') || e.target.closest('.settings-modal')) { start = null; return; }
        const t = e.touches[0];
        start = { x: t.clientX, y: t.clientY };
    }, { passive: true, capture: true });

    document.addEventListener('touchend', (e) => {
        if (!start) return;
        const t = e.changedTouches[0];
        const dx = t.clientX - start.x;
        const dy = t.clientY - start.y;
        start = null;

        if (Math.abs(dx) < MIN_DISTANCE) return;
        if (Math.abs(dy) > Math.abs(dx) * MAX_OFF_AXIS) return;

        // An open KB content card owns horizontal swipes: right returns to the
        // results list (mobile "back"). Consume the gesture either way so it
        // never falls through to a tab switch while reading.
        if (state.currentTab === 'chat' && state.conversationMode === 'read' && state.kbOpenItem) {
            if (dx > 0) executeAction('CLOSE_KB_ITEM');
            return;
        }

        const tabs = state.tabs;
        const n = tabs.length;
        const idx = tabs.findIndex(tab => tab.active);
        if (idx === -1 || n < 2) return;

        // Swipe left (dx<0) advances; swipe right goes back. The ribbon loops:
        // past the last tab wraps to the first and vice versa, so you can circle
        // round from either end.
        const next = ((dx < 0 ? idx + 1 : idx - 1) % n + n) % n;

        // Await the switch (render rebuilds the strip), then scroll the new
        // active tab into view so swiping toward the end keeps it visible.
        executeAction('SWITCH_TAB', tabs[next].id).then(snapActiveTabIntoView);
    }, { passive: true, capture: true });
}
