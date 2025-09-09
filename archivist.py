#!/usr/bin/env python3
"""Archive management script for build directory."""

import argparse
import os
import select
import shutil
import sys
import termios
import time
import tty
import zipfile
from pathlib import Path


def get_model_hash() -> str:
    """Get the model hash from the most recent history.log entry."""
    history_file = Path("history.log")
    
    if not history_file.exists():
        print("Error: history.log file does not exist.")
        print("Please run the model at least once to generate this file.")
        sys.exit(1)
    
    with open(history_file, "r") as f:
        first_line = f.readline().strip()
    
    if not first_line:
        print("Error: history.log is empty.")
        sys.exit(1)
    
    # Parse the hash from the history.log format: "timestamp | hash | command"
    parts = first_line.split(" | ")
    if len(parts) < 3:
        print("Error: history.log has unexpected format.")
        sys.exit(1)
    
    model_hash = parts[1].strip()
    return model_hash


def format_bytes(bytes_num):
    """Format bytes into human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_num) < 1024.0:
            return f"{bytes_num:3.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} PB"


def format_time(seconds):
    """Format seconds into human readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_progress_bar(current, total, start_time, prefix="Progress"):
    """Print a progress bar with percentage, data processed, and speed."""
    bar_length = 40
    filled_length = int(bar_length * current // total)

    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    percent = 100 * (current / total)

    elapsed = time.time() - start_time
    speed = current / elapsed if elapsed > 0 else 0

    # Create progress line
    progress_line = f"\r{prefix}: |{bar}| {percent:.1f}% "
    progress_line += f"[{format_bytes(current)}/{format_bytes(total)}] "
    progress_line += f"Speed: {format_bytes(speed)}/s"

    print(progress_line, end="", flush=True)

    if current >= total:
        print()  # New line when complete


class ProgressFileWrapper:
    """Wrapper around file objects to track read progress."""

    def __init__(self, file_obj, total_size, start_time, desc="Processing"):
        self.file_obj = file_obj
        self.total_size = total_size
        self.bytes_read = 0
        self.start_time = start_time
        self.desc = desc
        self.last_update = 0

    def read(self, size=-1):
        data = self.file_obj.read(size)
        self.bytes_read += len(data)

        # Update progress every MB or every 0.1 seconds
        current_time = time.time()
        if (
            self.bytes_read - self.last_update > 1024 * 1024
            or current_time - self.start_time > 0.1
        ):
            print_progress_bar(
                self.bytes_read, self.total_size, self.start_time, self.desc
            )
            self.last_update = self.bytes_read

        return data

    def __getattr__(self, name):
        return getattr(self.file_obj, name)


def save_project() -> None:
    """Create a zip file from the build directory and save it to archive using the model hash as name."""
    data_dir = Path("./build")
    archive_dir = Path("./staging")

    if not data_dir.exists():
        print(f"Error: Build directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Get the model hash as project name
    project_name = get_model_hash()

    # Create archive directory if it doesn't exist
    archive_dir.mkdir(exist_ok=True)

    # Create zip file path
    zip_path = archive_dir / f"{project_name}.zip"

    print(f"Saving project with hash: {project_name}")

    # Calculate total size to archive
    total_size = 0
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = Path(root) / file
            try:
                size = file_path.stat().st_size
                total_size += size
                file_list.append((file_path, size))
            except OSError:
                pass  # Skip files we can't stat

    # Create the zip file with progress tracking
    start_time = time.time()
    bytes_written = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zipf:
        for file_path, file_size in file_list:
            arcname = file_path.relative_to(data_dir.parent)

            # For large files, use streaming with progress
            if file_size > 10 * 1024 * 1024:  # Files larger than 10MB
                with open(file_path, "rb") as f:
                    # Create a ZipInfo manually to write in streaming mode
                    zinfo = zipfile.ZipInfo(filename=str(arcname))
                    zinfo.external_attr = 0o644 << 16  # Set file permissions
                    # Force ZIP64 format for large files
                    zinfo.file_size = file_size
                    zinfo.compress_size = (
                        file_size  # Since we're using STORED (no compression)
                    )

                    with zipf.open(zinfo, mode="w", force_zip64=True) as dest:
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            dest.write(chunk)
                            bytes_written += len(chunk)
                            print_progress_bar(
                                bytes_written, total_size, start_time, "Archiving"
                            )
            else:
                # For small files, use the regular write method
                zipf.write(file_path, arcname)
                bytes_written += file_size
                # Update progress for small files too
                print_progress_bar(bytes_written, total_size, start_time, "Archiving")

    # Final progress update
    print_progress_bar(total_size, total_size, start_time, "Archiving")

    elapsed = time.time() - start_time
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    speed_mb = size_mb / elapsed if elapsed > 0 else 0

    print(f"Archive: {zip_path}")
    print(f"Size: {format_bytes(zip_path.stat().st_size)}")
    print(f"Time: {format_time(elapsed)} (Speed: {speed_mb:.1f} MB/s)")


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

    archive_dir = Path("./staging")

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
            if key == "\x1b":  # Escape sequence for arrow keys
                # Read the next two characters
                key = sys.stdin.read(2)

                if key == "[A":  # Up arrow
                    selected = (selected - 1) % len(projects)
                elif key == "[B":  # Down arrow
                    selected = (selected + 1) % len(projects)
            elif key == "\r":  # Enter key
                # User made a selection
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print("\033[H\033[J", end="")  # Clear screen
                restore_project(projects[selected])
                return
            elif key.lower() == "q":  # Quit
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
    archive_dir = Path("./staging")
    data_dir = Path("./build")
    zip_path = archive_dir / f"{project_name}.zip"

    if not zip_path.exists():
        print(f"Error: Archive '{zip_path}' does not exist.")
        sys.exit(1)

    # Get archive size
    archive_size = zip_path.stat().st_size
    print(f"Restoring project '{project_name}' ({format_bytes(archive_size)})...")

    # Wipe the build directory if it exists
    if data_dir.exists():
        print("Clearing existing build directory...")
        shutil.rmtree(data_dir)

    # Create an empty build directory
    data_dir.mkdir(exist_ok=True)

    start_time = time.time()

    with zipfile.ZipFile(zip_path, "r", allowZip64=True) as zipf:
        # Get total uncompressed size
        total_size = sum(zinfo.file_size for zinfo in zipf.filelist)

        bytes_extracted = 0

        # Extract files with progress tracking
        for zinfo in zipf.filelist:
            # Extract file
            target_path = Path(".") / zinfo.filename

            # For large files, extract with streaming
            if zinfo.file_size > 10 * 1024 * 1024:  # Files larger than 10MB
                # Ensure target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)

                with zipf.open(zinfo) as source, open(target_path, "wb") as target:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        target.write(chunk)
                        bytes_extracted += len(chunk)
                        print_progress_bar(
                            bytes_extracted, total_size, start_time, "Extracting"
                        )

                # Set file permissions if available
                if hasattr(zinfo, "external_attr"):
                    # Extract Unix permissions from external_attr
                    unix_permissions = zinfo.external_attr >> 16
                    if unix_permissions:
                        os.chmod(target_path, unix_permissions)
            else:
                # For small files, use regular extract
                zipf.extract(zinfo, ".")
                bytes_extracted += zinfo.file_size
                print_progress_bar(
                    bytes_extracted, total_size, start_time, "Extracting"
                )

    # Final progress update
    print_progress_bar(total_size, total_size, start_time, "Extracting")

    elapsed = time.time() - start_time
    speed_mb = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"Restored: {project_name}")
    print(f"Extracted: {format_bytes(total_size)}")
    print(f"Time: {format_time(elapsed)} (Speed: {speed_mb:.1f} MB/s)")


def list_projects() -> None:
    """List all available zip files in the archive folder."""
    archive_dir = Path("./staging")

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
        description="Archive management for build directory"
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
    options_count = sum(
        [bool(args.save), bool(args.restore), bool(args.restore_id), args.list]
    )
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
