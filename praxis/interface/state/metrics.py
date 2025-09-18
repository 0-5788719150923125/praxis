"""Metrics state management for dashboard."""

from collections import deque
from datetime import datetime
from threading import Lock


class MetricsState:
    """Manages dashboard metrics and state."""

    def __init__(self, max_data_points=1000):
        self.lock = Lock()
        self.max_data_points = max_data_points

        # Training metrics
        self.train_losses = deque(maxlen=max_data_points)
        self.val_loss = None
        self.accuracy = None
        self.fitness = None
        self.memory_churn = None

        # Progress tracking
        self.batch = 0
        self.step = 0
        self.rate = 0
        self.num_tokens = 0
        self.context_tokens = 0

        # Model info
        self.total_params = "0M"
        self.local_experts = 0
        self.remote_experts = 0

        # Status
        self.mode = "train"
        self.status_text = "_initializing"
        self.url = "N/A"
        self.seed = None
        self.arg_hash = "000000"

        # Time tracking
        self.start_time = datetime.now()

        # Additional info
        self.info_dict = {}

        # Visual elements
        self.sign = 1

    def update_loss(self, train_loss):
        """Update training loss."""
        with self.lock:
            if train_loss is not None:
                self.train_losses.append(train_loss)

    def update_val(self, val_loss):
        """Update validation loss."""
        with self.lock:
            self.val_loss = val_loss

    def update_accuracy(self, acc0, acc1):
        """Update accuracy metrics."""
        with self.lock:
            self.accuracy = [acc0, acc1]

    def update_fitness(self, fitness):
        """Update fitness metric."""
        with self.lock:
            self.fitness = fitness

    def update_memory(self, churn):
        """Update memory churn metric."""
        with self.lock:
            self.memory_churn = churn

    def update_step(self, step):
        """Update current step."""
        with self.lock:
            self.step = step

    def update_batch(self, batch):
        """Update current batch."""
        with self.lock:
            self.batch = batch

    def update_rate(self, seconds):
        """Update processing rate."""
        with self.lock:
            self.rate = seconds

    def update_tokens(self, num_tokens):
        """Update token count."""
        with self.lock:
            self.num_tokens = num_tokens

    def update_context_tokens(self, context_tokens):
        """Update context token count."""
        with self.lock:
            self.context_tokens = context_tokens

    def update_expert_count(self, num_local, num_remote):
        """Update expert counts."""
        with self.lock:
            self.local_experts = num_local
            self.remote_experts = num_remote

    def update_url(self, url):
        """Update URL."""
        with self.lock:
            self.url = url

    def update_info(self, info_dict):
        """Update the key/value pairs to display in the info panel."""
        with self.lock:
            self.info_dict = {**self.info_dict, **info_dict}

    def update_status(self, status):
        """Update status text."""
        with self.lock:
            self.status_text = status

    def update_params(self, total_params):
        """Update total parameters."""
        with self.lock:
            reduced = int(total_params / 10**6)
            self.total_params = f"{reduced}M"

    def set_mode(self, mode):
        """Set current mode."""
        with self.lock:
            self.mode = mode

    def set_start_time(self, time):
        """Set start time."""
        with self.lock:
            self.start_time = time

    def update_seed(self, seed):
        """Update seed."""
        with self.lock:
            self.seed = seed

    def hours_since(self):
        """Calculate hours since start."""
        current_time = datetime.now()
        time_difference = current_time - self.start_time
        hours = time_difference.total_seconds() / 3600
        return hours
