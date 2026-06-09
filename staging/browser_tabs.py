#!/usr/bin/env python3
"""Enumerate open browser tabs via the Chrome DevTools Protocol and flag Praxis
instances. A feasibility probe for next/peer_bridge.md (local-browser discovery).

THE CATCH: this only sees tabs if the browser was launched with a debugging
port open, e.g.

    chromium --remote-debugging-port=9222
    google-chrome --remote-debugging-port=9222

Without that flag there is no list to read - the browser does not expose open
tabs to localhost by default. That opt-in is the real cost of this discovery
path, and the honest answer to "how hard is this": trivial to read, but it
requires the user to start their browser a specific way (or for us to relaunch
it). No silent inspection.

Usage:
    python browser_tabs.py [--ports 9222,9223] [--all]

By default prints only Praxis-looking tabs; --all prints everything found.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# Ports a Chromium browser commonly exposes when remote debugging is on. We scan
# a small range so multiple browser profiles / instances can be found at once.
DEFAULT_PORTS = list(range(9222, 9230))

# A tab is "Praxis-looking" if its URL or title matches any of these. Loose on
# purpose - localhost + a Praxis marker. Tighten to an origin allowlist before
# this becomes real discovery (see peer_bridge.md guardrails).
PRAXIS_HINTS = ("praxis", "localhost", "127.0.0.1", "0.0.0.0")


def fetch_tabs(port, timeout=0.4):
    """Return the DevTools tab list for one port, or [] if nothing answers."""
    url = f"http://127.0.0.1:{port}/json/list"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def is_praxis(tab):
    blob = f"{tab.get('url', '')} {tab.get('title', '')}".lower()
    return any(h in blob for h in PRAXIS_HINTS)


def origin_of(url):
    """scheme://host:port, the identity we actually pair on."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if not parts.scheme:
        return url
    return f"{parts.scheme}://{parts.netloc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", help="comma-separated debug ports to scan")
    ap.add_argument("--all", action="store_true", help="print every tab, not just Praxis")
    args = ap.parse_args()

    ports = (
        [int(p) for p in args.ports.split(",")] if args.ports else DEFAULT_PORTS
    )

    found, origins, any_port = [], set(), False
    for port in ports:
        tabs = fetch_tabs(port)
        if tabs:
            any_port = True
        for tab in tabs:
            if tab.get("type") != "page":
                continue  # skip service workers, devtools targets, etc.
            if args.all or is_praxis(tab):
                tab["_port"] = port
                found.append(tab)
                if is_praxis(tab):
                    origins.add(origin_of(tab.get("url", "")))

    if not any_port:
        print(
            "No debugging port answered on "
            f"{ports[0]}-{ports[-1]}.\n"
            "Launch your browser with --remote-debugging-port=9222 and retry.",
            file=sys.stderr,
        )
        return 1

    for tab in found:
        flag = "  [PRAXIS]" if is_praxis(tab) else ""
        print(f":{tab['_port']}{flag}  {origin_of(tab.get('url', ''))}")
        print(f"          title: {tab.get('title', '')!r}")
        print(f"          url:   {tab.get('url', '')}")
        print(f"          ws:    {tab.get('webSocketDebuggerUrl', '-')}")

    print(f"\nDistinct Praxis origins detected: {len(origins)}")
    for o in sorted(origins):
        print(f"  - {o}")
    if len(origins) >= 2:
        print("Two+ origins seen - a bridge between them is discoverable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
