import atexit
import json
import multiprocessing
import random
import subprocess
import time
from collections import deque
from queue import Empty, Full


class GunTransport:
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
            self._nodejs_process = subprocess.Popen(
                [
                    "/bin/bash",
                    "-c",
                    "npm install gun && node --experimental-network-imports transport/gun.mjs",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            print(f"Node.js server started with PID {self._nodejs_process.pid}")

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
                "Error: Node.js not found. Make sure Node.js is installed and in your PATH."
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

    def get_training_sample(self, num_entries=10):
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

        return [random.choice(["INPUT: ", "OUTPUT: "]) + entry for entry in sample]

    def __del__(self):
        if self._nodejs_process:
            print("Terminating Node.js process...")
            self._stop_event.set()
            self._nodejs_process.terminate()
            self._nodejs_process.wait()
            if self._process:
                self._process.join(timeout=5)
            print("Node.js process terminated.")


# Register the termination function to be called at exit
atexit.register(lambda: None)  # This ensures __del__ is called on program exit

if __name__ == "__main__":
    # Create an instance of GunTransport
    gun = GunTransport()

    # Retrieve the output whenever needed
    while True:
        time.sleep(10)
        output = gun.get_training_sample(1000)
        for i, item in enumerate(output):
            print(f"{i}: {item}")
