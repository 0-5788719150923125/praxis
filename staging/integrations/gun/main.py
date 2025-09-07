"""Gun.js integration implementation for Praxis."""

import atexit
import multiprocessing
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full
from typing import Any, Dict, Optional

from praxis.integrations.base import BaseIntegration, IntegrationSpec


class GunAdapter:
    """Manages connection to Gun.js server for decentralized chat data."""
    
    def __init__(self, max_cache_size=1000):
        self._nodejs_process = None
        self._output_queue = multiprocessing.Queue(maxsize=max_cache_size)
        self._stop_event = multiprocessing.Event()
        self._process = None
        self._max_cache_size = max_cache_size
        self._cache = deque(maxlen=max_cache_size)
        self._cache_lock = multiprocessing.Lock()
        self._connect_gun()

    def _connect_gun(self):
        try:
            # Get the path to the gun.mjs file in this module
            module_dir = Path(__file__).parent
            gun_mjs_path = module_dir / "gun.mjs"

            # Run npm install in the module directory if needed
            if not (module_dir / "node_modules").exists():
                print("[Gun] Installing Node.js dependencies...")
                npm_result = subprocess.run(
                    ["npm", "install"], cwd=module_dir, capture_output=True, text=True
                )
                if npm_result.returncode != 0:
                    print(f"[Gun] Warning: npm install failed: {npm_result.stderr}")

            self._nodejs_process = subprocess.Popen(
                ["node", "--experimental-network-imports", str(gun_mjs_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            print(f"[Gun] Node.js server started with PID {self._nodejs_process.pid}")

            self._process = multiprocessing.Process(
                target=self._read_nodejs_output,
                args=(
                    self._nodejs_process,
                    self._output_queue,
                    self._stop_event,
                    self._cache,
                    self._cache_lock,
                    self._max_cache_size,
                ),
            )
            self._process.start()

        except FileNotFoundError:
            print(
                "[Gun] Error: Node.js not found. Make sure Node.js is installed and in your PATH."
            )

    @staticmethod
    def _read_nodejs_output(
        process, queue, stop_event, cache, cache_lock, max_cache_size
    ):
        while not stop_event.is_set():
            line = process.stdout.readline()
            if not line:
                break
            event = line.strip()
            with cache_lock:
                cache.append(event)
                if len(cache) > max_cache_size:
                    cache.popleft()
            while True:
                try:
                    queue.put_nowait(event)
                    break
                except Full:
                    try:
                        queue.get_nowait()  # Remove oldest item if queue is full
                    except Empty:
                        pass  # Queue became empty, try putting again

    def get_sample(self, num_entries=10):
        new_entries = []
        while len(new_entries) < num_entries:
            try:
                entry = self._output_queue.get_nowait()
                new_entries.append(entry)
            except Empty:
                break

        with self._cache_lock:
            all_entries = list(self._cache) + new_entries
            self._cache.extend(new_entries)
            if len(self._cache) > self._max_cache_size:
                self._cache = deque(
                    all_entries[-self._max_cache_size :], maxlen=self._max_cache_size
                )
            sample = all_entries[-num_entries:]

        return sample

    def __del__(self):
        if self._nodejs_process:
            print("[Gun] Terminating Node.js process...")
            self._stop_event.set()
            self._nodejs_process.terminate()
            self._nodejs_process.wait()
            if self._process:
                self._process.join(timeout=5)
            print("[Gun] Node.js process terminated.")


# Legacy functions for backward compatibility
def add_cli_args(parser):
    """Add gun CLI arguments to the parser."""
    data_group = None

    # Find the 'data' argument group
    for group in parser._action_groups:
        if group.title == "data":
            data_group = group
            break

    if data_group is None:
        data_group = parser.add_argument_group("data")

    data_group.add_argument(
        "--gun",
        action="store_true",
        default=False,
        help="Use gun.js decentralized chat data for training",
    )


def initialize(args, cache_dir, ckpt_path=None, truncated_hash=None):
    """Initialize Gun module when conditions are met."""
    # Delegate to Integration class (will be handled by Integration.initialize)
    return {}


def provide_dataset(tokenizer, seed, config=None, *args):
    """Provide Gun dataset when requested."""
    # This will be handled by Integration.provide_dataset
    return None


def cleanup():
    """Cleanup Gun resources."""
    # This will be handled by Integration.cleanup
    pass


class Integration(BaseIntegration):
    """Gun.js integration for decentralized chat data."""

    def __init__(self, spec: IntegrationSpec):
        """Initialize the Gun integration."""
        super().__init__(spec)
        self.gun_adapter = None

    def add_cli_args(self, parser) -> None:
        """Add gun CLI arguments to the parser."""
        return add_cli_args(parser)

    def initialize(
        self, args: Any, cache_dir: str, ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initialize Gun module when conditions are met."""
        # Only initialize if the gun flag is actually set
        if not getattr(args, "gun", False):
            print("[Gun] Module not initialized (--gun flag not set)")
            return {}

        # Check if Node.js is available
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                print("[Gun] Error: Node.js not available")
                return {}
        except (subprocess.SubprocessError, FileNotFoundError):
            print(
                "[Gun] Error: Node.js not found. Please install Node.js to use Gun data source."
            )
            sys.exit(1)

        # Check for determinism warning
        if getattr(args, "seed", None) and not getattr(args, "dev", False):
            print(
                "WARNING: GUN chats are never deterministic, and cannot be reproduced when using a `seed`. "
                "You should omit the `--gun` argument for experiments."
            )
            time.sleep(5)

        # Create the Gun adapter
        self.gun_adapter = GunAdapter()
        
        # Mark as properly initialized
        self._initialized = True
        print("[Gun] Module initialized")
        return {}

    def cleanup(self) -> None:
        """Cleanup Gun resources."""
        if self.gun_adapter is not None:
            del self.gun_adapter
            self.gun_adapter = None

    def provide_dataset(
        self, tokenizer: Any, seed: int, config: Optional[Any] = None, *args
    ) -> Optional[Any]:
        """Provide Gun dataset when requested."""
        # Only provide dataset if properly initialized
        if not self._initialized or self.gun_adapter is None:
            print(
                "[Gun] ERROR: Dataset requested but module not initialized (--gun flag not set)"
            )
            return None

        # Import here to avoid circular dependency
        from builders import PraxisSampler

        class GunChatDataset(PraxisSampler):
            """Dataset class for Gun chat data."""

            def __init__(dataset_self, tokenizer):
                super().__init__(tokenizer)
                dataset_self.gun = self.gun_adapter  # Use the integration's adapter
                dataset_self.weight = 0.1

            def fill_sequence_cache(dataset_self):
                """Fill the sequence cache with Gun chat data."""
                # Get a list of text samples (these are individual messages)
                text_list = dataset_self.gun.get_sample(250)

                # Filter out empty messages
                messages = [msg.strip() for msg in text_list if msg and msg.strip()]

                if not messages:
                    # No real data available, return without adding anything
                    return

                # Create one continuous chat room conversation with all messages
                conversation = []

                # Add a system prompt describing the chat room context
                system_prompts = [
                    "You are participating in a decentralized chat network.",
                    "You are in a community chat room where multiple participants are having an ongoing discussion.",
                    "This is a free-form conversation between participants in a public chat.",
                    "You are observing and participating in a continuous stream of messages.",
                    "Write thy wrong.",
                ]
                conversation.append(
                    {"role": "system", "content": random.choice(system_prompts)}
                )

                # Assign alternating roles to simulate a continuous chat stream
                # Start with a random role for the first message
                current_role = random.choice(["user", "assistant"])

                for message in messages:
                    # Add each message with alternating roles
                    conversation.append({"role": current_role, "content": message})

                    # Alternate roles for next message to simulate multiple participants
                    current_role = "assistant" if current_role == "user" else "user"

                # Ensure conversation ends with assistant for proper training
                if conversation and conversation[-1]["role"] == "user":
                    # Remove the last message if it's from user
                    conversation.pop()

                # Only process if we have at least 2 messages (after potential pop)
                if len(conversation) >= 3:  # system + at least 2 messages
                    try:
                        # Format the entire conversation as one continuous sequence
                        formatted_text = dataset_self.tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=False
                        )
                        dataset_self.sequence_cache.append(formatted_text)
                    except Exception as e:
                        # Skip if tokenization fails
                        pass

        # Create instance with just the tokenizer (PraxisSampler signature)
        dataset = GunChatDataset(tokenizer)
        return dataset