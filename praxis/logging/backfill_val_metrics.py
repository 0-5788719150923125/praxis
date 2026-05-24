"""Null out redundant val_loss / val_perplexity rows in historical metrics.db files.

Older training runs logged val_* metrics on every training batch because
Lightning's callback_metrics retains them after validation ends. The result:
the same validation value sits on ~1000 consecutive rows, which flattens the
Research-tab charts and also trips the API's SQL sampling.

This script keeps the first row per contiguous run of equal values (the row
closest to when validation actually fired) and nulls the rest.

Usage:
    python -m praxis.logging.backfill_val_metrics [--dry-run] [--runs-dir PATH]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from praxis.metrics.training_metrics import validation_metric_names

# Derived from the training-metric registry: any metric flagged as
# ``is_validation`` gets the same consecutive-duplicate prune treatment.
VAL_COLUMNS = tuple(validation_metric_names())


def _count(cursor, col: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM metrics WHERE {col} IS NOT NULL")
    return cursor.fetchone()[0]


def backfill_db(db_path: Path, dry_run: bool = False) -> dict:
    """Null out repeat val_* values in one metrics.db. Returns per-column counts.

    SQLite in WAL mode needs write access to the containing directory even
    for read-only queries (it creates -shm/-wal files), so both dry-run and
    apply paths require a writable file. If the DBs are owned by root from a
    Docker run, chown them first or run this script as root.
    """
    result = {"path": str(db_path)}
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
        )
        if not cursor.fetchone():
            result["skipped"] = "no metrics table"
            return result

        # Trip wire for the permission-induced stale-view bug: if a -wal
        # file exists but sqlite opened read-only, the returned COUNT
        # reflects only the main .db and silently hides recent writes.
        wal = db_path.with_name(db_path.name + "-wal")
        if wal.exists() and wal.stat().st_size > 0:
            cursor.execute("SELECT COUNT(*) FROM metrics")
            if cursor.fetchone()[0] == 0:
                result["skipped"] = "WAL has data but view is empty (permissions?)"
                return result

        for col in VAL_COLUMNS:
            cursor.execute(f"SELECT COUNT(*) FROM metrics WHERE {col} IS NOT NULL")
            before = cursor.fetchone()[0]

            # Find steps whose value equals the previous non-null step's value.
            cursor.execute(f"""
                WITH ordered AS (
                    SELECT step, {col},
                           LAG({col}) OVER (ORDER BY step) AS prev_val
                    FROM metrics
                    WHERE {col} IS NOT NULL
                )
                SELECT step FROM ordered WHERE prev_val = {col}
                """)
            redundant_steps = [row[0] for row in cursor.fetchall()]

            if not dry_run and redundant_steps:
                # SQLite limits IN-list size, so batch the update.
                BATCH = 500
                for i in range(0, len(redundant_steps), BATCH):
                    chunk = redundant_steps[i : i + BATCH]
                    placeholders = ",".join("?" for _ in chunk)
                    cursor.execute(
                        f"UPDATE metrics SET {col} = NULL WHERE step IN ({placeholders})",
                        chunk,
                    )
                conn.commit()

            nulled = len(redundant_steps) if dry_run else before - _count(cursor, col)
            result[col] = {"before": before, "nulled": nulled}

        if not dry_run:
            cursor.execute("VACUUM")

    finally:
        conn.close()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("build/runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.runs_dir.exists():
        print(f"Runs dir not found: {args.runs_dir}", file=sys.stderr)
        return 1

    dbs = sorted(args.runs_dir.glob("*/metrics.db"))
    if not dbs:
        print(f"No metrics.db files under {args.runs_dir}")
        return 0

    mode = "DRY RUN" if args.dry_run else "APPLY"
    print(f"[{mode}] Processing {len(dbs)} metrics.db file(s)")

    total_nulled = {col: 0 for col in VAL_COLUMNS}
    for db in dbs:
        try:
            summary = backfill_db(db, dry_run=args.dry_run)
        except sqlite3.OperationalError as err:
            print(f"  {db}  ERROR: {err}")
            continue
        line_parts = [summary["path"]]
        for col in VAL_COLUMNS:
            stats = summary.get(col)
            if isinstance(stats, dict):
                line_parts.append(f"{col}: -{stats['nulled']}")
                total_nulled[col] += stats["nulled"]
        if "skipped" in summary:
            line_parts.append(f"skipped ({summary['skipped']})")
        print("  " + "  ".join(line_parts))

    summary_parts = [f"{col}=-{total_nulled[col]}" for col in VAL_COLUMNS]
    print("Total nulled: " + " ".join(summary_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
