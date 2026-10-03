#!/usr/bin/env python3
"""Capture a real web page as one tall PNG, for the tablet medium.

    capture.py <spec.json>

The spec is JSON - never argv, because ghost passes no free text on a command line (Windows
quoting): {"url", "out", "width", "height", "max_height", "user_agent"}. The page is opened in
headless Chromium (Playwright) at the tablet's width, given a moment to settle, and captured
from the top down to its own length, capped at max_height. Exit 0 and the PNG on success;
anything else is a failure, with the reason on stderr.
"""

import json
import sys

import re

from playwright.sync_api import sync_playwright

# THE POP-UPS A FIRST VISIT GETS - cookie consent, newsletter, "open in app" - answered the way a
# person would, most private answer first: a site's own button takes its own backdrop and scroll
# lock with it, which removing elements by force does not always manage.
DECLINE = re.compile(
    r"^\s*(accept (only )?(necessary|essential|required)( cookies)?( only)?|"
    r"(only|use) (necessary|essential|required)( cookies)?( only)?|"
    r"reject( all)?( cookies)?|decline( all)?|deny( all)?|refuse( all)?|"
    r"continue without accepting|necessary cookies only)\s*$",
    re.I,
)
DISMISS = re.compile(
    r"^\s*(accept( all)?( cookies)?|allow( all)?( cookies)?|i accept|agree|i agree|"
    r"got it|ok(ay)?|close|dismiss|no,? thanks|not now|maybe later|continue|×|✕)\s*$",
    re.I,
)

# Whatever is still pinned over the page and looks like a modal: a dialog, or a fixed layer
# covering a good part of the screen. Removed, and the page's scroll lock released.
CLEAR_OVERLAYS = """
() => {
  const vw = innerWidth, vh = innerHeight;
  let gone = 0;
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.position !== 'fixed' && cs.position !== 'sticky') continue;
    const r = el.getBoundingClientRect();
    const share = (Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0)) *
                   Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0))) / (vw * vh);
    const dialog = el.matches('[role=dialog],[role=alertdialog],[aria-modal=true],dialog') ||
                   el.querySelector('[role=dialog],[role=alertdialog],[aria-modal=true],dialog');
    // a site's own header bar is fixed too: only something covering a real share goes, or a
    // dialog of any size
    if (share > 0.25 || (dialog && share > 0.02)) { el.remove(); gone++; }
  }
  for (const el of [document.documentElement, document.body]) {
    if (!el) continue;
    el.style.setProperty('overflow', 'visible', 'important');
    el.style.setProperty('position', 'static', 'important');
  }
  return gone;
}
"""


# Where an answer may be clicked: inside something that IS a pop-up - never an ordinary "OK" or
# "Continue" on the page itself. A consent tool's own iframe counts as one whole.
POPUP = (
    "[role=dialog], [role=alertdialog], [aria-modal=true], dialog, "
    "[id*=consent i], [class*=consent i], [id*=cookie i], [class*=cookie i], "
    "[id*=gdpr i], [class*=gdpr i], [class*=modal i], [id*=modal i], [class*=popup i]"
)


def _click_first(page, pattern) -> bool:
    for frame in page.frames:
        try:
            scope = frame.locator(POPUP) if frame == page.main_frame else frame.locator("body")
            for role in ("button", "link"):
                loc = scope.get_by_role(role, name=pattern)
                for i in range(min(loc.count(), 6)):
                    el = loc.nth(i)
                    if el.is_visible():
                        el.click(timeout=2000)
                        return True
        except Exception:
            continue
    return False


def tidy(page) -> None:
    for _ in range(3):  # a second pop-up sometimes follows the first
        if not (_click_first(page, DECLINE) or _click_first(page, DISMISS)):
            break
        page.wait_for_timeout(700)
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass
    removed = page.evaluate(CLEAR_OVERLAYS)
    if removed:
        print(f"cleared {removed} overlay(s)")
    page.wait_for_timeout(400)


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    width = int(spec.get("width", 1200))
    height = int(spec.get("height", 1600))
    cap = int(spec.get("max_height", 6000))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
                user_agent=spec.get("user_agent") or None,
                locale="en-US",
            )
            page = ctx.new_page()
            page.goto(spec["url"], wait_until="domcontentloaded", timeout=45000)
            # late images and layout shifts; a page that never goes quiet still gets captured
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            page.wait_for_timeout(1500)
            if spec.get("tidy", True):
                tidy(page)
            full = page.evaluate(
                "Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)"
            )
            tall = max(height, min(int(full or height), cap))
            page.screenshot(
                path=spec["out"],
                full_page=True,
                clip={"x": 0, "y": 0, "width": width, "height": tall},
            )
        finally:
            browser.close()
    print(f"captured {spec['url']} -> {spec['out']} ({width}x{tall})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # the reason is the only thing ghost can show
        print(f"capture failed: {e}", file=sys.stderr)
        sys.exit(1)
