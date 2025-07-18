import atexit
import multiprocessing
import signal
import sys

from hivemind.hivemind_cli import run_dht


def run_dht_process():
    sys.argv = [sys.argv[0]] + [
        "--refresh_period",
        "30",
    ]  # Example: Set refresh period to 30 seconds
    run_dht.main()


def terminate_dht(dht_process):
    if dht_process.is_alive():
        print("Terminating DHT process...")
        dht_process.terminate()
        dht_process.join(timeout=5)
        if dht_process.is_alive():
            print("DHT process did not terminate gracefully, forcing...")
            dht_process.kill()


def start_dht_background():
    # Start the DHT process as a separate non-daemon process
    dht_process = multiprocessing.Process(target=run_dht_process, daemon=False)
    dht_process.start()

    # Register the termination function to be called when the main process exits
    atexit.register(terminate_dht, dht_process)

    return dht_process


def signal_handler(signum, frame):
    print(f"Received signal {signum}. Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the DHT in the background
    dht_process = start_dht_background()

    # Your main script continues here
    print("Main script is running...")
    print("DHT is running in the background.")

    try:
        # Example: do some work in the main thread
        import time

        while True:
            time.sleep(10)
            print("Main script is still running...")
    finally:
        print("Main script is exiting. Cleaning up...")
        terminate_dht(dht_process)

    print("Main script has exited.")
