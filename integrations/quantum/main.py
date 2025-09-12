"""Quantum code dataset implementation for Praxis."""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from praxis.data.formatters.files import format_file_as_messages
from praxis.data.config import SYSTEM_PROMPT
from praxis.integrations.base import BaseIntegration, IntegrationSpec

# Module state (for backward compatibility)
_quantum_enabled = False
_quantum_path = None


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add quantum module arguments to the parser."""
    parser.add_argument(
        "--quantum",
        action="store_true",
        help="Enable quantum code training data from qoblib repository",
    )


def initialize(args, cache_dir=None, ckpt_path=None, truncated_hash=None):
    """Initialize the quantum module."""
    global _quantum_enabled, _quantum_path

    # Check if quantum flag is set
    _quantum_enabled = getattr(args, "quantum", False)

    # Debug logging
    print(f"[Quantum] initialize() called with quantum flag: {_quantum_enabled}")
    if hasattr(args, "beta"):
        print(f"[Quantum] Beta flag is: {getattr(args, 'beta', False)}")

    if _quantum_enabled:
        # Get the project root directory (where run.py is located)
        # This works whether we're called from the module or from run.py
        module_path = Path(__file__).resolve()
        project_root = (
            module_path.parent.parent.parent
        )  # staging/quantum -> staging -> praxis root

        # Set up the build directory relative to project root
        data_dir = project_root / "build" / "quantum"
        data_dir.mkdir(parents=True, exist_ok=True)

        repo_path = data_dir / "qoblib"
        _quantum_path = repo_path

        # Check if repository exists and is valid
        if repo_path.exists():
            # Validate the existing repository
            try:
                # Check if it's a valid git repo
                result = subprocess.run(
                    ["git", "status"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    print(
                        f"[Quantum] Invalid repository detected, removing and re-cloning..."
                    )
                    import shutil

                    shutil.rmtree(repo_path)
                else:
                    # Check if sparse checkout is properly configured
                    sparse_check = subprocess.run(
                        ["git", "sparse-checkout", "list"],
                        cwd=str(repo_path),
                        capture_output=True,
                        text=True,
                    )
                    if sparse_check.returncode == 0 and sparse_check.stdout.strip():
                        # Sparse checkout is enabled, disable it to get all files
                        print(
                            "[Quantum] Disabling sparse checkout to access all files..."
                        )
                        subprocess.run(
                            ["git", "sparse-checkout", "disable"],
                            cwd=str(repo_path),
                            capture_output=True,
                        )
                    print(f"[Quantum] Using existing repository at {repo_path}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(
                    f"[Quantum] Repository validation failed, removing and re-cloning..."
                )
                import shutil

                shutil.rmtree(repo_path)

        # Clone the repository if it doesn't exist
        if not repo_path.exists():
            print("[Quantum] Cloning qoblib repository (this may take a while)...")
            try:
                # Try different URL formats - GitHub accepts both with and without .git
                urls_to_try = [
                    "https://github.com/Vectorrent/qoblib",  # Original format without .git
                    "git://github.com/Vectorrent/qoblib",  # git protocol without .git
                    "https://github.com/Vectorrent/qoblib.git",  # With .git as fallback
                ]

                clone_success = False
                last_error = None

                for url in urls_to_try:
                    try:
                        print(f"[Quantum] Attempting to clone from: {url}")

                        # Set up environment to prevent auth prompts
                        env = os.environ.copy()
                        env["GIT_TERMINAL_PROMPT"] = "0"  # Disable terminal prompts
                        env["GIT_ASKPASS"] = ""  # Empty askpass program
                        env["GCM_INTERACTIVE"] = (
                            "never"  # Disable Windows Git Credential Manager
                        )

                        result = subprocess.run(
                            [
                                "git",
                                "clone",
                                "--depth",
                                "1",  # Shallow clone to reduce size
                                "--single-branch",  # Only clone default branch
                                "--progress",  # Show progress
                                url,
                                str(repo_path),
                            ],
                            env=env,
                            stdin=subprocess.DEVNULL,  # Prevent input prompts
                            stderr=subprocess.PIPE,  # Capture stderr for progress
                            text=True,
                            timeout=300,  # 5 minute timeout for large repo
                            check=True,
                        )

                        clone_success = True
                        print("[Quantum] Repository cloned successfully")
                        break

                    except (
                        subprocess.CalledProcessError,
                        subprocess.TimeoutExpired,
                    ) as e:
                        last_error = e
                        if repo_path.exists():
                            import shutil

                            shutil.rmtree(repo_path)
                        continue

                if not clone_success:
                    raise (
                        last_error
                        if last_error
                        else Exception("Failed to clone repository")
                    )

                # Remove large archive files after cloning
                for pattern in ["*.tar.gz", "*.zip", "*.tar"]:
                    for archive in repo_path.rglob(pattern):
                        try:
                            archive.unlink()
                            print(f"[Quantum] Removed large file: {archive.name}")
                        except Exception:
                            pass

            except subprocess.TimeoutExpired:
                print("[Quantum] Error: Git clone timed out after 180 seconds")
                print(
                    "[Quantum] This repository contains large files and may be slow to clone"
                )
                print(
                    "[Quantum] You can try cloning manually: git clone --depth 1 --branch OUF_at_zurich.ibm.com-main-patch-70f8 https://github.com/Vectorrent/qoblib build/quantum/qoblib"
                )
                if repo_path.exists():
                    import shutil

                    shutil.rmtree(repo_path)
                _quantum_enabled = False
                return
            except subprocess.CalledProcessError as e:
                print(f"[Quantum] Error cloning repository: {e.stderr}")
                if repo_path.exists():
                    import shutil

                    shutil.rmtree(repo_path)
                _quantum_enabled = False
                return

        # Count available text files
        total_files = 0
        for ext in [
            ".py",
            ".md",
            ".txt",
            ".rst",
            ".sol",
            ".vrp",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".cfg",
            ".ini",
            ".sh",
            ".ipynb",
        ]:
            count = len(list(repo_path.rglob(f"*{ext}")))
            if count > 0:
                total_files += count
        print(f"[Quantum] Found {total_files} text files for training")


def _is_binary_file(file_path: Path) -> bool:
    """Check if a file appears to be binary."""
    try:
        with open(file_path, "rb") as f:
            # Read first 1024 bytes to check for binary content
            chunk = f.read(1024)
            # Check for null bytes (common in binary files)
            if b"\x00" in chunk:
                return True
            # Check if file is mostly non-text characters
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
            non_text = len([b for b in chunk if b not in text_chars])
            # If more than 30% non-text, consider it binary
            return non_text > len(chunk) * 0.3
    except Exception:
        return True  # If we can't read it, assume binary


def get_quantum_examples(num_examples: int = 10) -> List[Dict[str, str]]:
    """
    Get quantum code examples formatted as chat messages.

    Args:
        num_examples: Number of examples to return

    Returns:
        List of chat message dictionaries
    """
    if not _quantum_enabled or not _quantum_path:
        return []

    examples = []

    # Binary/archive extensions to skip
    SKIP_EXTENSIONS = {
        ".tar",
        ".gz",
        ".zip",
        ".bz2",
        ".xz",
        ".rar",
        ".7z",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
        ".pyc",
        ".pyo",
        ".pyd",
        ".egg",
        ".whl",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".h5",
        ".hdf5",
    }

    # Valid text file extensions to process
    VALID_EXTENSIONS = {
        ".py",  # Python code
        ".md",  # Markdown documentation
        ".txt",  # Text files
        ".rst",  # ReStructuredText
        ".sol",  # Solution files (routing solutions)
        ".vrp",  # Vehicle Routing Problem files
        ".yaml",  # YAML config
        ".yml",  # YAML config
        ".json",  # JSON data
        ".toml",  # TOML config
        ".cfg",  # Config files
        ".ini",  # INI config
        ".sh",  # Shell scripts
        ".ipynb",  # Jupyter notebooks
    }

    # Find all valid text files
    valid_files = []
    for ext in VALID_EXTENSIONS:
        pattern = f"*{ext}"
        for f in _quantum_path.rglob(pattern):
            # Skip test files
            if f.name.startswith("test_"):
                continue
            # Skip __pycache__ directories
            if "__pycache__" in str(f):
                continue
            # Skip files with binary/archive extensions in their path
            if any(bad_ext in str(f).lower() for bad_ext in SKIP_EXTENSIONS):
                continue
            valid_files.append(f)

    # Filter out large files and binary files
    filtered_files = []
    for f in valid_files:
        try:
            # Skip files larger than 5MB
            if f.stat().st_size > 5 * 1024 * 1024:
                continue
            # Skip binary files
            if _is_binary_file(f):
                continue
            filtered_files.append(f)
        except Exception:
            continue

    if not filtered_files:
        print("[Quantum] Warning: No valid text files found after filtering")
        return []

    # Sample random files
    sampled_files = random.sample(
        filtered_files, min(num_examples, len(filtered_files))
    )

    for file_path in sampled_files:
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip empty or very short files
            if len(content.strip()) < 50:
                continue

            # Get relative path from repo root
            relative_path = file_path.relative_to(_quantum_path)

            # Determine file type and format appropriately
            file_ext = file_path.suffix.lower()

            # Choose appropriate language tag for code blocks
            lang_map = {
                ".py": "python",
                ".sh": "bash",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".json": "json",
                ".toml": "toml",
                ".ini": "ini",
                ".cfg": "ini",
                ".md": "markdown",
                ".rst": "rst",
                ".sol": "text",  # Solution files
                ".vrp": "text",  # VRP problem files
                ".txt": "text",
                ".ipynb": "json",  # Jupyter notebooks are JSON
            }
            lang = lang_map.get(file_ext, "text")

            # Adjust the description based on file type
            if file_ext in [".md", ".rst", ".txt"]:
                file_desc = "documentation from the qoblib quantum computing library"
                user_prompt = (
                    f"Please summarize this {file_desc}:\n\n```{lang}\n{content}\n```"
                )
            elif file_ext == ".sol":
                file_desc = "quantum optimization solution data"
                user_prompt = (
                    f"Please explain this {file_desc}:\n\n```{lang}\n{content}\n```"
                )
            elif file_ext == ".vrp":
                file_desc = "Vehicle Routing Problem instance for quantum optimization"
                user_prompt = (
                    f"Please analyze this {file_desc}:\n\n```{lang}\n{content}\n```"
                )
            elif file_ext in [".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"]:
                file_desc = "configuration from the qoblib quantum computing library"
                user_prompt = (
                    f"Please explain this {file_desc}:\n\n```{lang}\n{content}\n```"
                )
            else:
                file_desc = "quantum computing code from the qoblib library"
                user_prompt = (
                    f"Please analyze this {file_desc}:\n\n```{lang}\n{content}\n```"
                )

            # Build messages with system, developer, user, and assistant
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "developer",
                    "content": f"Reading file: {relative_path}\nThis is {file_desc}. Study it to understand quantum algorithms and implementations.",
                },
                {"role": "user", "content": user_prompt},
            ]

            # Add the assistant's analysis
            analysis = _generate_quantum_analysis(content, file_path, file_ext)
            messages.append({"role": "assistant", "content": analysis})

            examples.append(
                {"messages": messages, "source": f"quantum:{relative_path}"}
            )

        except Exception as e:
            print(f"[Quantum] Error reading {file_path}: {e}")
            continue

    return examples


def _generate_quantum_analysis(code: str, file_path: Path, file_ext: str) -> str:
    """Generate a simple analysis of quantum content based on file type."""

    # Simple heuristic-based analysis
    analysis_parts = []

    # Handle different file types
    if file_ext in [".md", ".rst", ".txt"]:
        # Documentation files
        if "README" in str(file_path).upper():
            analysis_parts.append(
                f"This README file from {file_path} provides documentation for quantum computing components."
            )
        else:
            analysis_parts.append(
                f"This documentation from {file_path} explains quantum computing concepts and implementations."
            )
    elif file_ext == ".sol":
        # Solution files
        if "Route" in code:
            analysis_parts.append(
                "This solution file contains optimized routing paths from quantum optimization algorithms."
            )
        else:
            analysis_parts.append(
                "This file contains solution data from quantum optimization benchmarks."
            )
    elif file_ext == ".vrp":
        # Vehicle Routing Problem files
        analysis_parts.append(
            "This VRP instance defines a vehicle routing optimization problem suitable for quantum computing approaches."
        )
    elif file_ext in [".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"]:
        # Configuration files
        analysis_parts.append(
            f"This configuration file from {file_path} defines parameters for quantum computing experiments."
        )
    else:
        # Code files (Python, shell scripts, etc.)
        # Check for common quantum operations
        if "Hadamard" in code or "H(" in code:
            analysis_parts.append(
                "This code uses Hadamard gates to create superposition states."
            )

        if "CNOT" in code or "CX" in code:
            analysis_parts.append("It implements CNOT gates for quantum entanglement.")

        if "measure" in code.lower():
            analysis_parts.append("The code includes quantum measurement operations.")

        if "circuit" in code.lower():
            analysis_parts.append("This implements a quantum circuit.")

        if "qubit" in code.lower():
            analysis_parts.append(
                "The code works with qubits as the fundamental quantum unit."
            )

        # Check for specific algorithms
        if "grover" in code.lower():
            analysis_parts.append(
                "This appears to implement Grover's search algorithm."
            )
        elif "shor" in code.lower():
            analysis_parts.append(
                "This appears to implement Shor's factoring algorithm."
            )
        elif "qft" in code.lower() or "fourier" in code.lower():
            analysis_parts.append("This implements the Quantum Fourier Transform.")

    # Default if no specific patterns found
    if not analysis_parts:
        analysis_parts.append(
            f"This quantum computing content from {file_path} provides insights into quantum algorithms and implementations."
        )

    return " ".join(analysis_parts)


# Dataset weight configuration
QUANTUM_WEIGHT = 1.0


def provide_dataset(tokenizer, seed, config=None, *args):
    """Provide Quantum dataset when requested."""
    global _quantum_enabled

    # Debug logging
    print(f"[Quantum] provide_dataset() called, _quantum_enabled = {_quantum_enabled}")

    # Only provide dataset if properly initialized
    if not _quantum_enabled:
        print(
            "[Quantum] ERROR: Dataset requested but module not initialized (--quantum flag not set)"
        )
        return None

    from praxis.data.datasets import PraxisSampler

    class QuantumDataset(PraxisSampler):
        """Dataset class for Quantum code data."""

        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self.tokenizer = tokenizer
            self.weight = QUANTUM_WEIGHT
            self.examples_cache = []
            self.current_index = 0

        def get_document(self):
            """Get a quantum code document.

            Returns:
                Dictionary with messages and metadata
            """
            # Refill cache if empty
            if self.current_index >= len(self.examples_cache):
                self.examples_cache = get_quantum_examples(10)
                self.current_index = 0

                if not self.examples_cache:
                    # Try once more with just 1 example
                    self.examples_cache = get_quantum_examples(1)
                    if not self.examples_cache:
                        print("[Quantum] Warning: No quantum examples available")
                        return {"messages": [], "metadata": {}}

            # Get next example from cache
            if self.current_index < len(self.examples_cache):
                example = self.examples_cache[self.current_index]
                self.current_index += 1
                return {
                    "messages": example.get("messages", []),
                    "metadata": {
                        "source": example.get("source", "quantum:unknown"),
                        "format": "quantum_code",
                    },
                }

            return {"messages": [], "metadata": {}}

        def fill_sequence_cache(self):
            """Legacy method for compatibility - converts to old text format."""
            document_data = self.get_document()

            # Convert back to text for legacy compatibility
            if document_data and document_data.get("messages"):
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        document_data["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    self.sequence_cache.append(formatted)
                except Exception as e:
                    print(f"[Quantum] Error formatting example: {e}")

    # Create and return dataset instance
    dataset = QuantumDataset(tokenizer)
    return dataset


# Module metadata
__module_name__ = "quantum"
__module_version__ = "0.1.0"
__module_description__ = "Quantum computing code dataset from qoblib"


# Test section
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Simple test - just print an example
    class MockArgs:
        quantum = True

    # Initialize with mock args
    initialize(MockArgs())

    if not _quantum_enabled:
        print(
            "Failed to initialize quantum module - repository may need to be cloned manually"
        )
        print(
            "Run: git clone --depth 1 https://github.com/Vectorrent/qoblib build/quantum/qoblib"
        )
        sys.exit(1)

    # Get one example
    examples = get_quantum_examples(1)
    if not examples:
        print("No examples found in repository")
        sys.exit(1)

    example = examples[0]
    print(f"=== Sample Training Example from {example['source']} ===\n")

    # Print the chat template format manually
    for i, msg in enumerate(example["messages"]):
        if i > 0:
            print()  # Add newline between messages
        if msg["role"] == "system":
            print("[BOS]system")
            print(msg["content"])
        elif msg["role"] == "developer":
            print("[BOS]developer")
            print(msg["content"])
        elif msg["role"] == "user":
            print("[BOS]user")
            # Truncate long code in user message for display
            content = msg["content"]
            if len(content) > 1000:
                print(content[:1000] + "\n... [truncated for display]")
            else:
                print(content)
        elif msg["role"] == "assistant":
            print("[BOS]assistant")
            print(msg["content"])
        print("[SEP]", end="")


class Integration(BaseIntegration):
    """Quantum computing code dataset integration from qoblib."""

    def __init__(self, spec: IntegrationSpec):
        """Initialize the quantum integration."""
        super().__init__(spec)
        self.quantum_path = None
        self.quantum_enabled = False

    def add_cli_args(self, parser) -> None:
        """Add quantum module arguments to the parser."""
        return add_cli_args(parser)

    def initialize(
        self,
        args: Any,
        cache_dir: str,
        ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Initialize the quantum module."""
        global _quantum_enabled, _quantum_path

        # Call the legacy initialize function
        result = initialize(args, cache_dir, ckpt_path, truncated_hash)

        # Copy global state to instance variables
        self.quantum_enabled = _quantum_enabled
        self.quantum_path = _quantum_path

        if self.quantum_enabled:
            self._initialized = True

        return result

    def provide_dataset(
        self, tokenizer: Any, seed: int, config: Optional[Any] = None, *args
    ) -> Optional[Any]:
        """Provide quantum dataset when requested."""
        if not self.quantum_enabled or not self.quantum_path:
            return None

        # Use the legacy provide_dataset function
        return provide_dataset(tokenizer, seed, config, *args)
