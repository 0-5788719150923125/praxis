#!/usr/bin/env python3
"""Archive management script for data directory."""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
import termios
import tty
import select


def get_model_hash() -> str:
    """Read the model hash from MODEL_HASH.txt file."""
    hash_file_path = Path("./data/praxis/MODEL_HASH.txt")
    
    if not hash_file_path.exists():
        print(f"Error: Model hash file '{hash_file_path}' does not exist.")
        print("Please run the model at least once to generate this file.")
        sys.exit(1)
        
    with open(hash_file_path, "r") as f:
        model_hash = f.read().strip()
        
    return model_hash


def save_project() -> None:
    """Create a zip file from the data directory and save it to archive using the model hash as name."""
    data_dir = Path("./data")
    archive_dir = Path("./archive")

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Get the model hash as project name
    project_name = get_model_hash()

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

    print(f"Successfully saved project with hash '{project_name}' to {zip_path}")


def _get_key():
    """Wait for a keypress and return a single character string."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Check if there's input waiting, and if not, keep waiting
        while not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _get_available_projects():
    """Return a list of available projects."""
    archive_dir = Path("./archive")
    
    if not archive_dir.exists() or not list(archive_dir.glob("*.zip")):
        return []
        
    return sorted([zip_file.stem for zip_file in archive_dir.glob("*.zip")])


def _display_menu(options, selected=0):
    """Display the menu with the selected option highlighted."""
    # Clear the terminal
    print("\033[H\033[J", end="")
    
    print("Select a project to restore (use UP/DOWN arrows, ENTER to select):\n")
    
    for i, option in enumerate(options):
        if i == selected:
            # Highlight selected option (reverse video)
            print(f"\033[7m  {option}\033[0m")
        else:
            print(f"  {option}")
    
    print("\nPress 'q' to quit")


def interactive_restore():
    """Provide an interactive menu to select which project to restore."""
    projects = _get_available_projects()
    
    if not projects:
        print("No archived projects found.")
        return
    
    selected = 0
    
    # Save terminal state
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        # Display the initial menu
        _display_menu(projects, selected)
        
        while True:
            key = _get_key()
            
            # Check for navigation keys
            if key == '\x1b':  # Escape sequence for arrow keys
                # Read the next two characters
                key = sys.stdin.read(2)
                
                if key == '[A':  # Up arrow
                    selected = (selected - 1) % len(projects)
                elif key == '[B':  # Down arrow
                    selected = (selected + 1) % len(projects)
            elif key == '\r':  # Enter key
                # User made a selection
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print("\033[H\033[J", end="")  # Clear screen
                restore_project(projects[selected])
                return
            elif key.lower() == 'q':  # Quit
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print("\033[H\033[J", end="")  # Clear screen
                print("Restore cancelled.")
                return
            
            # Redisplay the menu with the new selection
            _display_menu(projects, selected)
    except Exception as e:
        # In case of error, restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print(f"Error: {e}")
        return


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
        action="store_true",
        help="Save project state using model hash as the archive name",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Interactively select a project to restore",
    )
    parser.add_argument(
        "--restore-id",
        type=str,
        metavar="project_name",
        help="Restore specific project from 'archive/<project_name>.zip'",
    )
    parser.add_argument("--list", action="store_true", help="List archived projects")

    args = parser.parse_args()

    # Check that exactly one option is specified
    options_count = sum([bool(args.save), bool(args.restore), bool(args.restore_id), args.list])
    if options_count == 0:
        parser.print_help()
        sys.exit(1)
    elif options_count > 1:
        print("Error: Only one option can be specified at a time.")
        sys.exit(1)

    if args.save:
        save_project()
    elif args.restore:
        interactive_restore()
    elif args.restore_id:
        restore_project(args.restore_id)
    elif args.list:
        list_projects()


if __name__ == "__main__":
    main()
