/**
 * Praxis Web - Mobile Optimizations
 * Clean tab carousel - clicked tabs float to left
 */

import { state } from './state.js';
import { executeAction } from './actions.js';

/**
 * Scroll the carousel so the active tab sits at the strip's left edge.
 * Queries fresh DOM each call (render() rebuilds the strip on tab switch).
 */
export function snapActiveTabIntoView() {
    const tabButtons = document.querySelector('.tab-buttons');
    if (!tabButtons || window.innerWidth > 768) return;
    const active = tabButtons.querySelector('.tab-button.active');
    if (!active) return;
    const offset = active.getBoundingClientRect().left - tabButtons.getBoundingClientRect().left;
    tabButtons.scrollBy({ left: offset, behavior: 'smooth' });
}

/**
 * Setup mobile tab carousel
 */
export function setupTabCarousel() {
    const tabButtons = document.querySelector('.tab-buttons');
    const tabNav = document.querySelector('.tab-nav');

    if (!tabButtons || !tabNav) return;

    // Only apply on mobile
    if (window.innerWidth > 768) return;

    function updateScrollIndicators() {
        const scrollLeft = tabButtons.scrollLeft;
        const maxScroll = tabButtons.scrollWidth - tabButtons.clientWidth;

        tabNav.classList.toggle('has-scroll-left', scrollLeft > 5);
        tabNav.classList.toggle('has-scroll-right', scrollLeft < maxScroll - 5);
    }

    // Scroll a tab so it sits at the container's left edge (fixed anchor)
    function scrollTabToLeft(button) {
        if (!button) return;
        const offset = button.getBoundingClientRect().left - tabButtons.getBoundingClientRect().left;
        tabButtons.scrollLeft += offset;
    }

    tabButtons.addEventListener('scroll', updateScrollIndicators);

    // Remember where a touch began so we can tell a tap from a scroll gesture
    let touchStart = null;

    tabButtons.addEventListener('touchstart', (e) => {
        const button = e.target.closest('.tab-button');
        touchStart = button ? { button, x: e.touches[0].clientX, y: e.touches[0].clientY } : null;
    }, { passive: true });

    tabButtons.addEventListener('touchend', (e) => {
        if (!touchStart) return;
        const { button, x, y } = touchStart;
        touchStart = null;

        // A finger that moved was scrolling the carousel, not pressing a tab
        const touch = e.changedTouches[0];
        if (Math.abs(touch.clientX - x) > 10 || Math.abs(touch.clientY - y) > 10) return;

        // Snap once, after the press has switched the active tab
        let snapped = false;
        const snap = () => {
            if (snapped || !button.classList.contains('active')) return;
            snapped = true;
            observer.disconnect();
            scrollTabToLeft(button);
        };
        const observer = new MutationObserver(snap);
        observer.observe(button, { attributes: true, attributeFilter: ['class'] });
        setTimeout(() => { observer.disconnect(); snap(); }, 50);
    }, { passive: true });

    // Fallback for non-touch devices (desktop)
    tabButtons.addEventListener('click', (e) => {
        if ('ontouchstart' in window) return;
        if (!e.target.closest('.tab-button')) return;
        requestAnimationFrame(() => scrollTabToLeft(tabButtons.querySelector('.tab-button.active')));
    });

    updateScrollIndicators();
    scrollTabToLeft(tabButtons.querySelector('.tab-button.active'));

    // Re-snap only on real width changes - the URL bar showing/hiding during a
    // scroll fires resize and would otherwise jerk the active tab to the left
    let lastWidth = window.innerWidth;
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768 || window.innerWidth === lastWidth) return;
        lastWidth = window.innerWidth;
        updateScrollIndicators();
        scrollTabToLeft(tabButtons.querySelector('.tab-button.active'));
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
