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

from playwright.sync_api import sync_playwright


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
