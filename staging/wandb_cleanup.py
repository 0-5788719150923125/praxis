#!/usr/bin/env python3
"""
WandB Run Cleanup Script

This script connects to a WandB workspace and deletes runs that have fewer than
a specified number of steps. It provides a dry-run mode for safety and asks for
confirmation before deleting runs.

Usage:
    python wandb_cleanup.py --entity unsafe --project praxis --min-steps 100
    python wandb_cleanup.py --entity unsafe --project praxis --min-steps 100 --dry-run
    python wandb_cleanup.py --entity unsafe --project praxis --min-steps 100 --no-confirm
"""

import argparse
import sys
from typing import List, Optional
import wandb
from wandb.apis.public import Run
from datetime import datetime
import time


def get_run_info(run: Run) -> dict:
    """Extract relevant information from a run."""
    try:
        # Get the step count - wandb tracks this in summary
        steps = run.summary.get("_step", 0) if run.summary else 0

        # Alternative ways to get step count if _step is not available
        if steps == 0 and run.history_keys:
            # Try to get the last step from history
            try:
                # This might be slow for runs with long history
                history = run.history()
                if not history.empty:
                    steps = len(history)
            except:
                pass

        return {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "steps": steps,
            "created_at": run.created_at,
            "tags": run.tags,
            "url": run.url,
        }
    except Exception as e:
        print(f"Error getting info for run {run.id}: {e}")
        return None


def find_runs_to_delete(
    entity: str, project: str, min_steps: int, state_filter: Optional[str] = None
) -> List[dict]:
    """
    Find all runs with fewer than min_steps.

    Args:
        entity: WandB entity/username
        project: WandB project name
        min_steps: Minimum number of steps to keep a run
        state_filter: Optional state filter ('finished', 'crashed', 'failed', 'running')

    Returns:
        List of run information dictionaries
    """
    print(f"Connecting to wandb.ai/{entity}/{project}...")
    api = wandb.Api()

    # Build filters
    filters = {}
    if state_filter:
        filters["state"] = state_filter

    try:
        runs = api.runs(f"{entity}/{project}", filters=filters)
        total_runs = len(runs)
        print(f"Found {total_runs} total runs in project")
    except Exception as e:
        print(f"Error accessing project: {e}")
        return []

    runs_to_delete = []

    print(f"Analyzing runs (this may take a while)...")
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"  Processed {i}/{total_runs} runs...")

        run_info = get_run_info(run)
        if run_info and run_info["steps"] < min_steps:
            runs_to_delete.append(run_info)

    print(f"Found {len(runs_to_delete)} runs with less than {min_steps} steps")
    return runs_to_delete


def delete_runs(
    entity: str, project: str, run_ids: List[str], delete_artifacts: bool = False
) -> int:
    """
    Delete the specified runs.

    Args:
        entity: WandB entity/username
        project: WandB project name
        run_ids: List of run IDs to delete
        delete_artifacts: Whether to delete associated artifacts

    Returns:
        Number of successfully deleted runs
    """
    api = wandb.Api()
    deleted_count = 0
    failed_count = 0

    for i, run_id in enumerate(run_ids):
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            run.delete(delete_artifacts=delete_artifacts)
            deleted_count += 1
            print(f"  [{i+1}/{len(run_ids)}] Deleted run: {run_id}")

            # Small delay to avoid rate limiting
            if i % 10 == 0 and i > 0:
                time.sleep(1)

        except Exception as e:
            failed_count += 1
            print(f"  [{i+1}/{len(run_ids)}] Failed to delete run {run_id}: {e}")

    if failed_count > 0:
        print(f"\nWarning: Failed to delete {failed_count} runs")

    return deleted_count


def format_run_table(runs: List[dict], max_rows: int = 20) -> str:
    """Format runs as a nice table for display."""
    if not runs:
        return "No runs to display"

    # Sort by steps (ascending) so runs with fewest steps appear first
    runs_sorted = sorted(runs, key=lambda x: x["steps"])

    # Truncate if too many
    display_runs = runs_sorted[:max_rows]
    truncated = len(runs_sorted) > max_rows

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"{'Run Name':<30} {'State':<12} {'Steps':<10} {'Created':<20}")
    lines.append("-" * 80)

    for run in display_runs:
        name = run["name"][:29] if len(run["name"]) > 29 else run["name"]
        created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        created_str = created.strftime("%Y-%m-%d %H:%M")

        lines.append(
            f"{name:<30} {run['state']:<12} {run['steps']:<10} {created_str:<20}"
        )

    if truncated:
        lines.append(f"... and {len(runs_sorted) - max_rows} more runs")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Delete WandB runs with fewer than a specified number of steps"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="unsafe",
        help="WandB entity/username (default: unsafe)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="praxis",
        help="WandB project name (default: praxis)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        required=True,
        help="Minimum number of steps to keep a run",
    )
    parser.add_argument(
        "--state",
        type=str,
        choices=["finished", "crashed", "failed", "running"],
        help="Only consider runs in this state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="Also delete artifacts associated with the runs",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum number of runs to display in the table (default: 20)",
    )

    args = parser.parse_args()

    # Find runs to delete
    print(f"\nSearching for runs with less than {args.min_steps} steps...")
    if args.state:
        print(f"Filtering for runs in state: {args.state}")

    runs_to_delete = find_runs_to_delete(
        args.entity, args.project, args.min_steps, args.state
    )

    if not runs_to_delete:
        print("No runs found matching the criteria.")
        return 0

    # Display the runs that will be deleted
    print(format_run_table(runs_to_delete, args.max_display))

    print(f"\nTotal runs to delete: {len(runs_to_delete)}")

    if args.dry_run:
        print("\n[DRY RUN] No runs were actually deleted.")
        return 0

    # Confirm deletion
    if not args.no_confirm:
        print(
            f"\nYou are about to DELETE {len(runs_to_delete)} runs from {args.entity}/{args.project}"
        )
        if args.delete_artifacts:
            print("WARNING: This will also delete associated artifacts!")

        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            print("Deletion cancelled.")
            return 1

    # Perform deletion
    print(f"\nDeleting {len(runs_to_delete)} runs...")
    run_ids = [run["id"] for run in runs_to_delete]
    deleted_count = delete_runs(
        args.entity, args.project, run_ids, args.delete_artifacts
    )

    print(f"\nSuccessfully deleted {deleted_count} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
