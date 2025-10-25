/**
 * Praxis Web - Mobile Optimizations
 * Tab carousel and mobile-specific behavior
 */

/**
 * Setup mobile tab carousel with scroll indicators
 */
export function setupTabCarousel() {
    const tabButtons = document.querySelector('.tab-buttons');
    const tabNav = document.querySelector('.tab-nav');

    if (!tabButtons || !tabNav) return;

    // Only apply on mobile
    if (window.innerWidth > 768) return;

    let scrollEndTimer = null;

    function updateScrollIndicators() {
        const scrollLeft = tabButtons.scrollLeft;
        const scrollWidth = tabButtons.scrollWidth;
        const clientWidth = tabButtons.clientWidth;

        // Update scroll indicators
        if (scrollLeft > 5) {
            tabNav.classList.add('has-scroll-left');
        } else {
            tabNav.classList.remove('has-scroll-left');
        }

        if (scrollLeft < scrollWidth - clientWidth - 5) {
            tabNav.classList.add('has-scroll-right');
        } else {
            tabNav.classList.remove('has-scroll-right');
        }
    }

    function snapToNearestButton() {
        const buttons = tabButtons.querySelectorAll('.tab-button');
        const containerRect = tabButtons.getBoundingClientRect();
        const containerCenter = containerRect.left + containerRect.width / 2;

        let nearestButton = null;
        let minDistance = Infinity;

        buttons.forEach(button => {
            const rect = button.getBoundingClientRect();
            const buttonCenter = rect.left + rect.width / 2;
            const distance = Math.abs(buttonCenter - containerCenter);

            if (distance < minDistance) {
                minDistance = distance;
                nearestButton = button;
            }
        });

        if (nearestButton) {
            nearestButton.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
        }
    }

    // Update indicators on scroll
    tabButtons.addEventListener('scroll', () => {
        updateScrollIndicators();

        // Snap to nearest button after scrolling stops
        clearTimeout(scrollEndTimer);
        scrollEndTimer = setTimeout(() => {
            snapToNearestButton();
        }, 150);
    });

    // Initial state
    updateScrollIndicators();

    // Center active button on load
    const activeButton = tabButtons.querySelector('.tab-button.active');
    if (activeButton) {
        setTimeout(() => {
            activeButton.scrollIntoView({ behavior: 'auto', inline: 'center', block: 'nearest' });
        }, 100);
    }

    // Re-calculate on window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            updateScrollIndicators();
        }
    });
}
