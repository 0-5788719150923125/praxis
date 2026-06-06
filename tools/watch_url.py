"""Manage the spider's watchlist (build/spider.db).

Usage:
    python tools/watch_url.py                       # list watched sites
    python tools/watch_url.py https://example.com   # watch a site
    python tools/watch_url.py --remove https://example.com
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.spider import SpiderSettings
from praxis.spider.store import SpiderStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Site URL to watch")
    parser.add_argument("--remove", metavar="URL", help="Stop watching a site")
    parser.add_argument(
        "--max-sites",
        type=int,
        default=SpiderSettings().max_sites,
        help="Watchlist cap when adding (default: %(default)s)",
    )
    args = parser.parse_args()

    store = SpiderStore()
    if args.remove:
        store.remove_site(args.remove)
        print(f"Stopped watching {args.remove}")
    elif args.url:
        added = store.add_site(args.url, args.max_sites)
        print(f"{'Watching' if added else 'Already watching'} {args.url}")
    else:
        sites = store.list_sites()
        if not sites:
            print("No watched sites. The spider seeds from links in docs/ and next/.")
        for url, last_fetch, streak, enabled, pages in sites:
            status = "ok" if enabled else f"disabled ({streak} errors)"
            ago = (
                f"{int((time.time() - last_fetch) / 60)}m ago"
                if last_fetch
                else "never"
            )
            print(f"{url}  pages={pages}  last_fetch={ago}  {status}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
