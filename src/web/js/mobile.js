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
        const scrollWidth = tabButtons.scrollWidth;
        const clientWidth = tabButtons.clientWidth;

        // Show left fade if scrolled right
        if (scrollLeft > 5) {
            tabNav.classList.add('has-scroll-left');
        } else {
            tabNav.classList.remove('has-scroll-left');
        }

        // Show right fade if more content to the right
        if (scrollLeft < scrollWidth - clientWidth - 5) {
            tabNav.classList.add('has-scroll-right');
        } else {
            tabNav.classList.remove('has-scroll-right');
        }
    }

    // Scroll clicked/active tab to left-most position (fixed anchor)
    function scrollTabToLeft(button) {
        if (!button) return;

        // Get fresh positions - button should already have .active class
        const containerRect = tabButtons.getBoundingClientRect();
        const buttonRect = button.getBoundingClientRect();

        // Calculate distance from button to container's left edge
        const offset = buttonRect.left - containerRect.left;

        // Instant scroll to exact position (no smooth animation)
        tabButtons.scrollLeft = tabButtons.scrollLeft + offset;
    }

    // Update indicators on scroll
    tabButtons.addEventListener('scroll', () => {
        updateScrollIndicators();
    });

    // Track which tab was touched
    let touchedButton = null;

    // Track touchstart - just remember which button
    tabButtons.addEventListener('touchstart', (e) => {
        touchedButton = e.target.closest('.tab-button');
    }, { passive: true });

    // On touchend - wait for the clicked button to become active, then scroll it
    tabButtons.addEventListener('touchend', (e) => {
        if (!touchedButton) return;

        const clickedButton = touchedButton;
        touchedButton = null;

        // Use MutationObserver to wait for the active class to actually change
        const observer = new MutationObserver(() => {
            if (clickedButton.classList.contains('active')) {
                observer.disconnect();
                scrollTabToLeft(clickedButton);
            }
        });

        // Watch for class changes on the clicked button
        observer.observe(clickedButton, {
            attributes: true,
            attributeFilter: ['class']
        });

        // Fallback timeout in case the button was already active
        setTimeout(() => {
            observer.disconnect();
            if (clickedButton.classList.contains('active')) {
                scrollTabToLeft(clickedButton);
            }
        }, 50);
    }, { passive: true });

    // Fallback for non-touch devices (desktop)
    tabButtons.addEventListener('click', (e) => {
        // Only handle click if not a touch device (avoid double-trigger)
        if (!('ontouchstart' in window)) {
            const button = e.target.closest('.tab-button');
            if (button) {
                requestAnimationFrame(() => {
                    const activeButton = tabButtons.querySelector('.tab-button.active');
                    if (activeButton) {
                        scrollTabToLeft(activeButton);
                    }
                });
            }
        }
    });

    // Initial state
    updateScrollIndicators();

    // Scroll active button to left on load
    const activeButton = tabButtons.querySelector('.tab-button.active');
    if (activeButton) {
        scrollTabToLeft(activeButton);
    }

    // Re-calculate on window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            updateScrollIndicators();
            const active = tabButtons.querySelector('.tab-button.active');
            if (active) scrollTabToLeft(active);
        }
    });
}
