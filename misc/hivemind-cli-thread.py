import subprocess
import sys
from threading import Thread

from hivemind.hivemind_cli import run_dht


def run_dht_process():
    run_dht.main()


def main():
    # Start the DHT process in a separate thread
    dht_thread = Thread(target=run_dht_process, daemon=True)
    dht_thread.start()

    try:
        # Your main script logic here
        print("Main script is running...")
        while True:
            # Do your main script work here
            pass
    except KeyboardInterrupt:
        print("Main script interrupted, shutting down...")
    finally:
        # The DHT process will be terminated when the main script exits
        # because it's running in a daemon thread
        pass


if __name__ == "__main__":
    main()
