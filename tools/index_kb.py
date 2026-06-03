#!/usr/bin/env python3
"""Build the Praxis knowledge-base search index from on-disk sources.

Usage:
    python tools/index_kb.py                  # full rebuild, all sources
    python tools/index_kb.py --sources docs runs
    python tools/index_kb.py --query "rope theta"   # rebuild then test a query
    python tools/index_kb.py --db build/kb.db

Reads the corpus directly (docs/, next/, build/runs/) and writes an FTS5
index the web app serves at /api/kb/search. Re-run after the corpus changes;
the background indexer/spider will eventually call KBIndex.upsert() instead.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.kb import DEFAULT_DB_PATH, KB_SOURCE_REGISTRY, KBIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(KB_SOURCE_REGISTRY),
        help="Sources to index (default: all).",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Index DB path.")
    parser.add_argument("--query", help="Run a test search after rebuilding.")
    parser.add_argument(
        "--limit", type=int, default=10, help="Test-query result count."
    )
    args = parser.parse_args()

    index = KBIndex(db_path=Path(args.db))
    count = index.rebuild(sources=args.sources)
    print(f"Indexed {count} items into {args.db}")

    if args.query:
        print(f"\nTop {args.limit} for {args.query!r}:")
        for hit in index.search(args.query, limit=args.limit):
            print(f"  [{hit.item.type}] {hit.item.title}  (score {hit.score:.2f})")
            if hit.snippet:
                print(f"      {hit.snippet}")
    index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
