"""Test tab header spacing and layout."""

from playwright.sync_api import Page


def test_research_tab_header_spacing(page: Page):
    """Inspect spacing between tab navigation and tab header in Research tab."""

    # Navigate to app
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)

    # Click Research tab
    research_button = page.locator('button[data-tab="research"]')
    if research_button.count() > 0:
        research_button.click()
        page.wait_for_timeout(2000)  # Wait for content to load

        # Get elements
        tab_buttons = page.locator('.tab-buttons')
        tab_nav = page.locator('.tab-nav')
        research_content = page.locator('#research-content')
        tab_header = page.locator('.tab-header')

        print("\n" + "="*80)
        print("TAB HEADER SPACING ANALYSIS - RESEARCH TAB")
        print("="*80)

        # Check if elements exist
        print(f"\n✓ .tab-buttons exists: {tab_buttons.count() > 0}")
        print(f"✓ .tab-nav exists: {tab_nav.count() > 0}")
        print(f"✓ #research-content exists: {research_content.count() > 0}")
        print(f"✓ .tab-header exists: {tab_header.count() > 0}")

        if tab_buttons.count() > 0:
            tab_buttons_box = tab_buttons.bounding_box()
            print(f"\n.tab-buttons bounding box: {tab_buttons_box}")

            # Get computed styles
            tab_buttons_styles = page.evaluate("""() => {
                const el = document.querySelector('.tab-buttons');
                const styles = window.getComputedStyle(el);
                return {
                    marginBottom: styles.marginBottom,
                    paddingBottom: styles.paddingBottom,
                    height: styles.height
                };
            }""")
            print(f".tab-buttons computed styles:")
            for key, value in tab_buttons_styles.items():
                print(f"  {key}: {value}")

        if tab_nav.count() > 0:
            tab_nav_box = tab_nav.bounding_box()
            print(f"\n.tab-nav bounding box: {tab_nav_box}")

            tab_nav_styles = page.evaluate("""() => {
                const el = document.querySelector('.tab-nav');
                const styles = window.getComputedStyle(el);
                return {
                    marginBottom: styles.marginBottom,
                    paddingBottom: styles.paddingBottom,
                    borderBottom: styles.borderBottom
                };
            }""")
            print(f".tab-nav computed styles:")
            for key, value in tab_nav_styles.items():
                print(f"  {key}: {value}")

        if research_content.count() > 0:
            research_content_box = research_content.bounding_box()
            print(f"\n#research-content bounding box: {research_content_box}")

            research_content_styles = page.evaluate("""() => {
                const el = document.getElementById('research-content');
                const styles = window.getComputedStyle(el);
                return {
                    paddingTop: styles.paddingTop,
                    marginTop: styles.marginTop
                };
            }""")
            print(f"#research-content computed styles:")
            for key, value in research_content_styles.items():
                print(f"  {key}: {value}")

        if tab_header.count() > 0:
            tab_header_box = tab_header.bounding_box()
            print(f"\n.tab-header bounding box: {tab_header_box}")

            tab_header_styles = page.evaluate("""() => {
                const el = document.querySelector('.tab-header');
                const styles = window.getComputedStyle(el);
                return {
                    paddingTop: styles.paddingTop,
                    paddingBottom: styles.paddingBottom,
                    marginTop: styles.marginTop,
                    marginBottom: styles.marginBottom,
                    top: styles.top,
                    position: styles.position
                };
            }""")
            print(f".tab-header computed styles:")
            for key, value in tab_header_styles.items():
                print(f"  {key}: {value}")

        # Calculate the gap
        if 'tab_header_box' in locals() and tab_nav_box and tab_header_box:
            tab_nav_bottom = tab_nav_box['y'] + tab_nav_box['height']
            tab_header_top = tab_header_box['y']
            gap = tab_header_top - tab_nav_bottom

            print(f"\n{'='*80}")
            print(f"GAP ANALYSIS:")
            print(f"  .tab-nav bottom edge: {tab_nav_bottom}px")
            print(f"  .tab-header top edge: {tab_header_top}px")
            print(f"  GAP: {gap}px")
            print(f"{'='*80}\n")

            if gap > 1:
                print(f"⚠ ISSUE: Gap of {gap}px detected between tab nav and header")
                print(f"   Expected: 0px (no whitespace)")
            else:
                print(f"✓ No gap detected")

        print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        test_research_tab_header_spacing(page)
        browser.close()
