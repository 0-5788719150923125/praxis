"""Multi-directory file dataset."""

import os
import random
from itertools import cycle
from typing import Dict, List, Optional, Set

from transformers import PreTrainedTokenizer

from praxis.data.config import DEVELOPER_PROMPTS, SYSTEM_PROMPT
from praxis.data.datasets.base import PraxisSampler
from praxis.data.formatters.files import format_file_as_messages


class MultiDirectoryDataset(PraxisSampler):
    """Dataset that reads files from multiple directories."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        directories: List[str],
        allowed_extensions: Optional[List[str]] = [],
        excluded_dirs: Optional[List[str]] = None,
    ):
        super().__init__(tokenizer)
        # Normalize and resolve all directory paths
        self.cwd = os.getcwd()
        self.directories = []
        for d in directories:
            # If path is absolute, use it as-is; otherwise make it relative to CWD
            if os.path.isabs(d):
                normalized = os.path.normpath(d)
            else:
                normalized = os.path.normpath(os.path.join(self.cwd, d))
            self.directories.append(normalized)

        # Remove root directory if accidentally included
        if "/" in self.directories:
            self.directories.remove("/")
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]

        # Default exclusions for common development directories
        default_exclusions = {
            ".git",
            ".venv",
            "__pycache__",
            "staging",
            "build",
            "dist",
            "node_modules",
            "praxis.egg-info",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
        }
        user_exclusions = set(excluded_dirs) if excluded_dirs else set()
        self.excluded_dirs = default_exclusions.union(user_exclusions)

        print(f"[DATA] Working directory: {self.cwd}")
        print(f"[DATA] Scanning directories: {self.directories}")

        self.file_list = self._get_file_list()
        print(f"Found {len(self.file_list)} files")
        random.shuffle(self.file_list)
        self.file_iterator = iter(self.file_list)
        # Track files that have been removed due to read errors
        self.removed_files: Set[str] = set()

    def _should_skip_directory(self, dirpath: str) -> bool:
        """
        Check if directory should be skipped based on exclusion rules.
        """
        dir_name = os.path.basename(dirpath)

        # Check if directory is in excluded list
        if dir_name in self.excluded_dirs:
            return True

        # For absolute paths outside CWD, don't skip (e.g., temp directories)
        # Only apply CWD restriction for relative paths that were resolved
        try:
            real_path = os.path.realpath(dirpath)
            # If the original directory was absolute and outside CWD, allow it
            for orig_dir in self.directories:
                if real_path.startswith(orig_dir):
                    return False
            # Otherwise check if it's within CWD
            if real_path.startswith(self.cwd):
                return False
            return True
        except:
            # If there's any error resolving the path, skip it to be safe
            return True

    def _get_file_list(self) -> List[str]:
        """
        Recursively traverse directories and return a flat list of fully-qualified file paths,
        staying within the working directory context.
        """
        all_files = []

        for directory in self.directories:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist, skipping...")
                continue

            try:
                # Walk through directory recursively
                for root, dirs, files in os.walk(
                    directory, topdown=True, followlinks=False
                ):
                    # Modify dirs in-place to prevent walking into excluded directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if not self._should_skip_directory(os.path.join(root, d))
                    ]

                    for filename in files:
                        # Get full path
                        full_path = os.path.join(root, filename)

                        # For files, check if they're within allowed directories
                        real_path = os.path.realpath(full_path)
                        path_ok = False
                        for orig_dir in self.directories:
                            if real_path.startswith(orig_dir):
                                path_ok = True
                                break
                        if not path_ok and real_path.startswith(self.cwd):
                            path_ok = True
                        if not path_ok:
                            continue

                        # Check if file extension is allowed
                        file_ext = os.path.splitext(filename)[1].lower()
                        if len(self.allowed_extensions) > 0:
                            if file_ext in self.allowed_extensions:
                                all_files.append(full_path)
                        else:
                            all_files.append(full_path)
            except Exception as e:
                print(f"Error scanning directory {directory}: {str(e)}")
                continue

        return all_files

    def _read_file(self, file_path: str) -> Optional[str]:
        """Read and return the contents of a file, or None if it fails."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception:
                # Silently fail and return None to indicate removal needed
                return None
        except (FileNotFoundError, PermissionError, OSError):
            # File doesn't exist or can't be read - silently remove it
            return None
        except Exception:
            # Any other error - silently remove the file from list
            return None

    def get_document(self) -> Dict:
        """Get a formatted document with messages and metadata."""
        from praxis.data.config import SYSTEM_PROMPT

        # Get a file and its content
        max_attempts = len(self.file_list) + 10
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            try:
                # Get next file
                file_path = next(self.file_iterator)

                # Skip if we've already removed this file
                if file_path in self.removed_files:
                    continue

                # Try to read the file
                content = self._read_file(file_path)

                if content is None:
                    # File couldn't be read - remove it from the list silently
                    self.removed_files.add(file_path)
                    continue

                # Get relative path for formatting
                rel_path = os.path.relpath(file_path, self.cwd)

                # Format as messages
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Continue or complete the following code from {rel_path}:",
                    },
                    {"role": "assistant", "content": content},
                ]

                result = {
                    "messages": messages,
                    "metadata": {"source": "multi_dir", "file_path": file_path},
                }
                return result

            except StopIteration:
                # No more files, reset the iterator
                self.file_iterator = cycle(self.file_list)
                continue

        # Fallback - return empty document
        return {"messages": [], "metadata": {}}

    def fill_sequence_cache(self):
        """Fill the sequence cache with formatted file content."""
        max_attempts = len(self.file_list) + 10  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            try:
                # Get next file
                file_path = next(self.file_iterator)

                # Skip if we've already removed this file
                if file_path in self.removed_files:
                    continue

                # Try to read the file
                content = self._read_file(file_path)

                if content is None:
                    # File couldn't be read - remove it from the list silently
                    self.removed_files.add(file_path)
                    self.file_list = [
                        f for f in self.file_list if f not in self.removed_files
                    ]
                    continue

                # Skip empty files
                if not content.strip():
                    continue

                # Build a proper message structure
                try:
                    # Start with system and developer prompts
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "developer",
                            "content": DEVELOPER_PROMPTS.get(
                                "continue_text",
                                "Continue or complete the provided text.",
                            ),
                        },
                    ]

                    # Add the file content as assistant message
                    file_messages = format_file_as_messages(
                        file_path=file_path, content=content
                    )
                    messages.extend(file_messages)

                    # Apply chat template to format the messages
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )

                    self.sequence_cache.append(formatted_text)
                    return  # Successfully added to cache

                except Exception:
                    # If formatting fails, fall back to simple format
                    simple_content = self.tokenizer.bos_token + content
                    self.sequence_cache.append(simple_content)
                    return

            except StopIteration:
                # End of file list - reshuffle and start over
                # Remove any files that couldn't be read
                self.file_list = [
                    f for f in self.file_list if f not in self.removed_files
                ]

                if not self.file_list:
                    # No files left to read
                    print(
                        "Warning: No readable files remaining in MultiDirectoryDataset"
                    )
                    # Add a dummy entry to prevent crashes
                    self.sequence_cache.append(
                        self.tokenizer.bos_token + "No files available."
                    )
                    return

                random.shuffle(self.file_list)
                self.file_iterator = iter(self.file_list)
                continue

        # Fallback if we hit max attempts
        print(
            "Warning: Max attempts reached in MultiDirectoryDataset.fill_sequence_cache"
        )
        self.sequence_cache.append(self.tokenizer.bos_token + "Error loading files.")
        self.sequence_cache.append(self.tokenizer.bos_token + "Error loading files.")
