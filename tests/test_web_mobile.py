"""Web UI mobile viewport tests using Playwright."""

import re

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure mobile viewport for tests - Pixel 6 size."""
    return {
        **browser_context_args,
        "viewport": {
            "width": 412,  # Pixel 6 width
            "height": 915,
        },
        "device_scale_factor": 2.625,
        "is_mobile": True,
        "has_touch": True,
    }


def test_mobile_viewport_no_horizontal_overflow(page: Page):
    """Test that no elements overflow horizontally on mobile."""
    # Navigate to app (assuming it's running locally)
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)
    
    # Get viewport and document widths
    viewport_width = page.viewport_size["width"]
    
    # Check body doesn't overflow
    body_scroll_width = page.evaluate("() => document.body.scrollWidth")
    assert body_scroll_width <= viewport_width + 5, f"Body scrollWidth {body_scroll_width} exceeds viewport {viewport_width}"
    
    # Check each tab
    tabs = ["chat", "terminal", "agents", "research", "spec"]
    
    for tab_id in tabs:
        # Click tab button
        tab_button = page.locator(f'button[data-tab="{tab_id}"]')
        if tab_button.count() > 0:
            tab_button.click()
            page.wait_for_timeout(500)
            
            # Check tab container doesn't overflow
            tab_container = page.locator(f'#{tab_id}-container, .{tab_id}-container').first
            if tab_container.count() > 0:
                box = tab_container.bounding_box()
                if box:
                    right_edge = box["x"] + box["width"]
                    assert right_edge <= viewport_width, f"Tab {tab_id} overflows: right edge {right_edge} > viewport {viewport_width}"


def test_mobile_input_field_visible(page: Page):
    """Test that input field is visible and properly sized on mobile."""
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)
    
    # Get viewport width
    viewport_width = page.viewport_size["width"]
    
    # Check input field
    input_field = page.locator("#message-input")
    expect(input_field).to_be_visible()
    
    box = input_field.bounding_box()
    assert box is not None, "Input field has no bounding box"
    
    # Check input doesn't overflow
    right_edge = box["x"] + box["width"]
    assert right_edge <= viewport_width, f"Input field overflows: right edge {right_edge} > viewport {viewport_width}"
    
    # Check input is at least 200px wide
    assert box["width"] >= 200, f"Input field too narrow: {box['width']}px"


def test_mobile_chart_cards_fit_viewport(page: Page):
    """Test that chart cards don't overflow on mobile."""
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)
    
    # Navigate to Research tab
    research_button = page.locator('button[data-tab="research"]')
    if research_button.count() > 0:
        research_button.click()
        page.wait_for_timeout(1000)
        
        # Check chart cards
        chart_cards = page.locator(".chart-card")
        count = chart_cards.count()
        
        viewport_width = page.viewport_size["width"]
        
        for i in range(min(count, 3)):  # Check first 3 charts
            card = chart_cards.nth(i)
            box = card.bounding_box()
            if box:
                right_edge = box["x"] + box["width"]
                assert right_edge <= viewport_width + 5, f"Chart card {i} overflows: {right_edge} > {viewport_width}"


def test_mobile_code_blocks_fit_viewport(page: Page):
    """Test that code blocks don't overflow on mobile."""
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)
    
    # Navigate to Identity tab
    spec_button = page.locator('button[data-tab="spec"]')
    if spec_button.count() > 0:
        spec_button.click()
        page.wait_for_timeout(1000)
        
        # Check code blocks
        code_blocks = page.locator(".spec-code, pre")
        count = code_blocks.count()
        
        viewport_width = page.viewport_size["width"]
        
        for i in range(min(count, 3)):
            block = code_blocks.nth(i)
            box = block.bounding_box()
            if box:
                right_edge = box["x"] + box["width"]
                assert right_edge <= viewport_width + 5, f"Code block {i} overflows: {right_edge} > {viewport_width}"


def test_mobile_all_tabs_accessible(page: Page):
    """Test that all tabs are accessible on mobile."""
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)

    tabs = ["chat", "terminal", "agents", "research", "spec"]

    for tab_id in tabs:
        button = page.locator(f'button[data-tab="{tab_id}"]')
        if button.count() > 0:
            button.click()
            page.wait_for_timeout(300)

            # Verify tab is active (button has both 'tab-button' and 'active' classes)
            expect(button).to_have_class(re.compile(r".*active.*"))
