"""Quantum code dataset integration class for Praxis."""

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from praxis.integrations.base import BaseIntegration, IntegrationSpec


class Integration(BaseIntegration):
    """Quantum computing code dataset integration from qoblib."""

    def __init__(self, spec: IntegrationSpec):
        """Initialize the quantum integration."""
        super().__init__(spec)
        self.quantum_path = None
        self.quantum_enabled = False

    def add_cli_args(self, parser) -> None:
        """Add quantum module arguments to the parser."""
        parser.add_argument(
            "--quantum",
            action="store_true",
            help="Enable quantum code training data from qoblib repository",
        )

    def initialize(
        self, args: Any, cache_dir: str, ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initialize the quantum module."""
        # Check if quantum flag is set
        self.quantum_enabled = getattr(args, "quantum", False)

        if self.quantum_enabled:
            # Get the project root directory (where run.py is located)
            # This works whether we're called from the module or from run.py
            module_path = Path(__file__).resolve()
            project_root = (
                module_path.parent.parent.parent.parent
            )  # integration.py -> quantum -> integrations -> staging -> praxis root

            # Set up the build directory relative to project root
            data_dir = project_root / "build" / "quantum"
            data_dir.mkdir(parents=True, exist_ok=True)

            repo_path = data_dir / "qoblib"
            self.quantum_path = repo_path

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
                    # Clone with progress output
                    result = subprocess.run(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--single-branch",
                            "https://github.com/Vectorrent/qoblib",
                            str(repo_path),
                        ],
                        text=True,
                        timeout=180,  # 180 second timeout
                        check=True,
                    )
                    print("[Quantum] Repository cloned successfully")

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
                        "[Quantum] You can try cloning manually: git clone --depth 1 https://github.com/Vectorrent/qoblib build/quantum/qoblib"
                    )
                    if repo_path.exists():
                        import shutil
                        shutil.rmtree(repo_path)
                    self.quantum_enabled = False
                    return {}
                except subprocess.CalledProcessError as e:
                    print(f"[Quantum] Error cloning repository: {e}")
                    if repo_path.exists():
                        import shutil
                        shutil.rmtree(repo_path)
                    self.quantum_enabled = False
                    return {}

            self._initialized = True
            print("[Quantum] Module initialized")

        return {}

    def provide_dataset(
        self, tokenizer: Any, seed: int, config: Optional[Any] = None, *args
    ) -> Optional[Any]:
        """Provide quantum dataset when requested."""
        if not self.quantum_enabled or not self.quantum_path:
            return None

        # Import here to avoid circular dependency
        from builders import PraxisSampler

        class QuantumCodeDataset(PraxisSampler):
            """Dataset for quantum computing code samples."""

            def __init__(dataset_self, tokenizer):
                super().__init__(tokenizer)
                dataset_self.quantum_path = self.quantum_path
                dataset_self.weight = 0.2
                dataset_self.extensions = {".cirq", ".py", ".qasm", ".qs", ".json"}
                dataset_self._load_code_files()

            def _load_code_files(dataset_self):
                """Load all quantum code files from the repository."""
                dataset_self.code_files = []
                
                # Skip certain directories that don't contain useful code
                skip_dirs = {".git", "__pycache__", ".pytest_cache", "docs", "tests"}
                
                for root, dirs, files in os.walk(dataset_self.quantum_path):
                    # Remove directories we want to skip
                    dirs[:] = [d for d in dirs if d not in skip_dirs]
                    
                    for file in files:
                        # Check if file has a relevant extension
                        if any(file.endswith(ext) for ext in dataset_self.extensions):
                            file_path = Path(root) / file
                            dataset_self.code_files.append(file_path)
                
                print(f"[Quantum] Found {len(dataset_self.code_files)} quantum code files")

            def fill_sequence_cache(dataset_self):
                """Fill the sequence cache with quantum code samples."""
                if not dataset_self.code_files:
                    return

                # Sample a subset of files
                num_samples = min(100, len(dataset_self.code_files))
                sampled_files = random.sample(dataset_self.code_files, num_samples)

                for file_path in sampled_files:
                    try:
                        # Read the code file
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            code = f.read()

                        if not code.strip():
                            continue

                        # Create a conversation format for the code
                        # Get relative path for context
                        rel_path = file_path.relative_to(dataset_self.quantum_path)
                        
                        # Choose a random system prompt
                        system_prompts = [
                            "You are an expert in quantum computing and quantum algorithms.",
                            "You are a quantum software engineer proficient in quantum circuit design.",
                            "You specialize in quantum programming languages and frameworks.",
                            "You are learning quantum computing by studying code examples.",
                        ]
                        
                        # Choose a random instruction style
                        instruction_styles = [
                            f"Here's a quantum computing implementation from {rel_path}:",
                            f"Study this quantum code from the file {rel_path}:",
                            f"Analyze the following quantum algorithm implementation:",
                            f"This is an example of quantum programming in {file_path.suffix} format:",
                        ]

                        conversation = [
                            {"role": "system", "content": random.choice(system_prompts)},
                            {"role": "user", "content": random.choice(instruction_styles)},
                            {"role": "assistant", "content": code},
                        ]

                        # Format as conversation
                        formatted_text = dataset_self.tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=False
                        )
                        dataset_self.sequence_cache.append(formatted_text)

                    except Exception as e:
                        # Skip files that can't be read or processed
                        continue

        return QuantumCodeDataset(tokenizer)