#!/usr/bin/env python3
"""Archive management script for data directory."""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


def save_project(project_name: str) -> None:
    """Create a zip file from the data directory and save it to archive."""
    data_dir = Path("./data")
    archive_dir = Path("./archive")

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Create archive directory if it doesn't exist
    archive_dir.mkdir(exist_ok=True)

    # Create zip file path
    zip_path = archive_dir / f"{project_name}.zip"

    # Create the zip file (using STORED for no compression)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(data_dir.parent)
                zipf.write(file_path, arcname)

    print(f"Successfully saved project '{project_name}' to {zip_path}")


def restore_project(project_name: str) -> None:
    """Restore a zip file back to the data directory."""
    archive_dir = Path("./archive")
    data_dir = Path("./data")
    zip_path = archive_dir / f"{project_name}.zip"

    if not zip_path.exists():
        print(f"Error: Archive '{zip_path}' does not exist.")
        sys.exit(1)

    # Wipe the data directory if it exists
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"Cleared existing data directory: {data_dir}")
    
    # Create an empty data directory
    data_dir.mkdir(exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(".")

    print(f"Successfully restored project '{project_name}' from {zip_path}")


def list_projects() -> None:
    """List all available zip files in the archive folder."""
    archive_dir = Path("./archive")

    if not archive_dir.exists():
        print("No archive directory found.")
        return

    zip_files = list(archive_dir.glob("*.zip"))

    if not zip_files:
        print("No archived projects found.")
        return

    print("Available projects:")
    for zip_file in sorted(zip_files):
        project_name = zip_file.stem
        print(f"  - {project_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Archive management for data directory"
    )

    parser.add_argument(
        "--save",
        type=str,
        metavar="project_name",
        help="Save project state to 'archive/<project_name>.zip'",
    )
    parser.add_argument(
        "--restore",
        type=str,
        metavar="project_name",
        help="Restore project from 'archive/<project_name>.zip'",
    )
    parser.add_argument("--list", action="store_true", help="List archived projects")

    args = parser.parse_args()

    # Check that exactly one option is specified
    options_count = sum([bool(args.save), bool(args.restore), args.list])
    if options_count == 0:
        parser.print_help()
        sys.exit(1)
    elif options_count > 1:
        print("Error: Only one option can be specified at a time.")
        sys.exit(1)

    if args.save:
        save_project(args.save)
    elif args.restore:
        restore_project(args.restore)
    elif args.list:
        list_projects()


if __name__ == "__main__":
    main()
