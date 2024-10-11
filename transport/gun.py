import atexit
import json
import random
import subprocess
import threading
import time


class GunTransport:
    def __init__(self):
        self._nodejs_process = None
        self._output_list = []
        self._output_lock = threading.Lock()
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

            # Start a thread to read the output
            threading.Thread(
                target=self._read_nodejs_output,
                args=(self._nodejs_process,),
                daemon=True,
            ).start()

        except FileNotFoundError:
            print(
                "Error: Node.js not found. Make sure Node.js is installed and in your PATH."
            )

    def _read_nodejs_output(self, process):
        for line in process.stdout:
            event = line.strip()
            with self._output_lock:
                self._output_list.append(event)
                while len(self._output_list) > 20:
                    self._output_list.pop(0)

    def get_nodejs_output(self):
        with self._output_lock:
            return self._output_list.copy()

    def get_training_sample(self, num_entries=10):
        with self._output_lock:
            recent_entries = self._output_list[-num_entries:]
            return [
                random.choice(["INPUT: ", "OUTPUT: "]) + entry
                for entry in recent_entries
            ]

    def __del__(self):
        if self._nodejs_process:
            print("Terminating Node.js process...")
            self._nodejs_process.terminate()
            self._nodejs_process.wait()
            print("Node.js process terminated.")


# Register the termination function to be called at exit
atexit.register(lambda: None)  # This ensures __del__ is called on program exit

if __name__ == "__main__":
    # Create an instance of GunConnector
    gun = GunTransport()

    # Retrieve the output whenever needed
    while True:
        time.sleep(10)
        # output = gun.get_nodejs_output()
        output = gun.get_training_sample(1000)
        for i, item in enumerate(output):
            print(f"{i}: {item}")
