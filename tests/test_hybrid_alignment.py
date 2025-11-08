"""Test hybrid mode layout alignment between light and dark themes."""

import pytest
from playwright.sync_api import Page, expect


def test_hybrid_mode_alignment(page: Page):
    """Verify that light and dark theme elements are perfectly aligned in hybrid mode."""
    # Navigate to app
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)

    # Force hybrid mode activation by toggling theme until it activates
    max_attempts = 20
    hybrid_active = False

    for _ in range(max_attempts):
        # Toggle theme
        theme_button = page.locator(".theme-toggle-button")
        theme_button.click()
        page.wait_for_timeout(300)

        # Check if hybrid mode is active
        hybrid_overlay = page.locator(".hybrid-overlay")
        if hybrid_overlay.count() > 0 and hybrid_overlay.is_visible():
            hybrid_active = True
            break

    assert hybrid_active, "Failed to activate hybrid mode after 20 attempts"

    # Wait for hybrid to stabilize
    page.wait_for_timeout(1000)

    # Get bounding boxes of key elements in both layers
    original_app = page.locator(".app-container").first
    hybrid_app = page.locator(".hybrid-overlay .app-container").first

    # Verify both exist
    expect(original_app).to_be_visible()
    expect(hybrid_app).to_be_visible()

    # Get their positions and dimensions
    original_box = original_app.bounding_box()
    hybrid_box = hybrid_app.bounding_box()

    assert original_box is not None, "Original app-container not found"
    assert hybrid_box is not None, "Hybrid app-container not found"

    # Verify width alignment (should be identical)
    assert (
        abs(original_box["width"] - hybrid_box["width"]) < 2
    ), f"Width mismatch: original={original_box['width']}, hybrid={hybrid_box['width']}"

    # Verify both containers have proper max-width constraint
    original_width = page.evaluate(
        "() => getComputedStyle(document.querySelector('.app-container')).maxWidth"
    )
    hybrid_width = page.evaluate(
        "() => getComputedStyle(document.querySelector('.hybrid-overlay .app-container')).maxWidth"
    )

    assert (
        original_width == hybrid_width
    ), f"max-width mismatch: original={original_width}, hybrid={hybrid_width}"

    # Verify text content alignment by checking first message in each
    original_msg = page.locator(".app-container .message").first
    hybrid_msg = page.locator(".hybrid-overlay .message").first

    if original_msg.count() > 0 and hybrid_msg.count() > 0:
        orig_msg_box = original_msg.bounding_box()
        hyb_msg_box = hybrid_msg.bounding_box()

        if orig_msg_box and hyb_msg_box:
            # X positions should be very close (within 2px)
            assert (
                abs(orig_msg_box["x"] - hyb_msg_box["x"]) < 2
            ), f"Message X position mismatch: original={orig_msg_box['x']}, hybrid={hyb_msg_box['x']}"

            # Y positions should be very close (within 2px)
            assert (
                abs(orig_msg_box["y"] - hyb_msg_box["y"]) < 2
            ), f"Message Y position mismatch: original={orig_msg_box['y']}, hybrid={hyb_msg_box['y']}"


def test_hybrid_overlay_fully_opaque(page: Page):
    """Verify hybrid overlay has full opacity (no transparency)."""
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)

    # Force hybrid mode
    for _ in range(20):
        page.locator(".theme-toggle-button").click()
        page.wait_for_timeout(300)

        if page.locator(".hybrid-overlay").count() > 0:
            break

    page.wait_for_timeout(500)

    # Check opacity
    overlay = page.locator(".hybrid-overlay").first
    if overlay.count() > 0:
        opacity = page.evaluate(
            "() => getComputedStyle(document.querySelector('.hybrid-overlay')).opacity"
        )
        assert (
            opacity == "1"
        ), f"Hybrid overlay should be fully opaque, got opacity={opacity}"
