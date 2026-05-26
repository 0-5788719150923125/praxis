/**
 * Praxis Web - Mobile Optimizations
 * Clean tab carousel - clicked tabs float to left
 */

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
