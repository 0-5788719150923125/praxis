"""
Performance monitoring for tracking FPS and latency of pipeline stages.
Helps identify bottlenecks in real-time processing.
"""

import time
from collections import defaultdict, deque


class PerformanceMonitor:
    """Monitor performance metrics for video processing pipeline."""

    def __init__(self, window_size=100):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.last_fps_count = 0

    def measure(self, name):
        """
        Context manager for measuring execution time of a code block.

        Args:
            name: Name of the stage being measured

        Returns:
            Timer: Context manager for timing

        Example:
            with monitor.measure('detection'):
                landmarks = detector.detect(frame)
        """
        return self.Timer(self, name)

    class Timer:
        """Context manager for timing code blocks."""

        def __init__(self, monitor, name):
            self.monitor = monitor
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
            self.monitor.timings[self.name].append(elapsed)

    def record_frame(self):
        """Record that a frame has been processed."""
        self.frame_count += 1

    def get_fps(self):
        """
        Calculate current FPS.

        Returns:
            float: Frames per second
        """
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed < 0.1:  # Update at most every 100ms
            return 0.0

        frames_processed = self.frame_count - self.last_fps_count
        fps = frames_processed / elapsed

        self.last_fps_time = current_time
        self.last_fps_count = self.frame_count

        return fps

    def get_average_timing(self, name):
        """
        Get average timing for a specific stage.

        Args:
            name: Stage name

        Returns:
            float: Average time in milliseconds, or 0 if no data
        """
        if name not in self.timings or len(self.timings[name]) == 0:
            return 0.0

        return sum(self.timings[name]) / len(self.timings[name])

    def get_total_latency(self):
        """
        Get total pipeline latency (sum of all stages).

        Returns:
            float: Total latency in milliseconds
        """
        total = 0.0
        for name in self.timings:
            total += self.get_average_timing(name)
        return total

    def print_stats(self, detailed=False):
        """
        Print performance statistics.

        Args:
            detailed: If True, print per-stage timings
        """
        fps = self.get_fps()
        total_latency = self.get_total_latency()

        print(f"\n{'='*60}")
        print(f"Performance Statistics (Frame {self.frame_count})")
        print(f"{'='*60}")
        print(f"FPS: {fps:.1f}")
        print(f"Total Latency: {total_latency:.2f}ms")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")

        if detailed and self.timings:
            print(f"\n{'Stage':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
            print(f"{'-'*60}")

            for name in sorted(self.timings.keys()):
                times = list(self.timings[name])
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    print(f"{name:<20} {avg_time:>10.2f}   {min_time:>10.2f}   {max_time:>10.2f}")

        print(f"{'='*60}\n")

    def should_print_stats(self, interval_frames=100):
        """
        Check if it's time to print stats.

        Args:
            interval_frames: Print stats every N frames

        Returns:
            bool: True if stats should be printed
        """
        return self.frame_count % interval_frames == 0 and self.frame_count > 0

    def get_stats_dict(self):
        """
        Get performance statistics as dictionary.

        Returns:
            dict: Dictionary with performance metrics
        """
        stats = {
            'fps': self.get_fps(),
            'total_latency_ms': self.get_total_latency(),
            'frame_count': self.frame_count,
            'runtime_s': time.time() - self.start_time,
            'stages': {}
        }

        for name in self.timings:
            times = list(self.timings[name])
            if times:
                stats['stages'][name] = {
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times)
                }

        return stats

    def reset(self):
        """Reset all performance counters."""
        self.timings.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.last_fps_count = 0
