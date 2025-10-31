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

    // Scroll clicked/active tab to left-most position
    function scrollTabToLeft(button) {
        if (!button) return;

        // Scroll to start (left-most position)
        button.scrollIntoView({ behavior: 'smooth', inline: 'start', block: 'nearest' });
    }

    // Update indicators on scroll
    tabButtons.addEventListener('scroll', () => {
        updateScrollIndicators();
    });

    // Intercept tab clicks to scroll to left
    tabButtons.addEventListener('click', (e) => {
        const button = e.target.closest('.tab-button');
        if (button) {
            // Scroll clicked tab to left after a brief delay (allows state update)
            setTimeout(() => scrollTabToLeft(button), 50);
        }
    });

    // Initial state
    updateScrollIndicators();

    // Scroll active button to left on load
    const activeButton = tabButtons.querySelector('.tab-button.active');
    if (activeButton) {
        setTimeout(() => {
            scrollTabToLeft(activeButton);
        }, 100);
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
