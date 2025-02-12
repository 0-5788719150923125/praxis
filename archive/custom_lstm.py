import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.W_ii = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate
        self.W_if = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # Cell gate
        self.W_ig = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

        # Output gate
        self.W_io = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

        # Set forget gate bias to 1 to help with learning long sequences
        nn.init.constant_(self.b_f, 1.0)

    def forward(self, x, state=None):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (h, c) each of shape (batch_size, hidden_size)
                  If None, initialize with zeros
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        batch_size = x.size(0)

        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = state

        # Input gate
        i = torch.sigmoid(x @ self.W_ii.t() + h @ self.W_hi.t() + self.b_i)

        # Forget gate
        f = torch.sigmoid(x @ self.W_if.t() + h @ self.W_hf.t() + self.b_f)

        # Cell gate
        g = torch.tanh(x @ self.W_ig.t() + h @ self.W_hg.t() + self.b_g)

        # Output gate
        o = torch.sigmoid(x @ self.W_io.t() + h @ self.W_ho.t() + self.b_o)

        # New cell state
        c_next = f * c + i * g

        # New hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def forward_sequence(self, x_sequence):
        """
        Process a sequence of inputs

        Args:
            x_sequence: Input tensor of shape (seq_len, batch_size, input_size)
        Returns:
            outputs: Tensor of shape (seq_len, batch_size, hidden_size)
            (h_n, c_n): Final states
        """
        seq_len, batch_size, _ = x_sequence.size()
        outputs = []
        state = None

        for t in range(seq_len):
            h, c = self.forward(x_sequence[t], state)
            outputs.append(h)
            state = (h, c)

        return torch.stack(outputs), state


def test_single_step():
    print("\nTesting single step forward pass...")

    # Initialize model
    input_size, hidden_size = 10, 20
    lstm = SimpleLSTM(input_size, hidden_size)

    # Test single input
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    h, c = lstm(x)

    # Check shapes
    assert h.shape == (
        batch_size,
        hidden_size,
    ), f"Expected hidden state shape {(batch_size, hidden_size)}, got {h.shape}"
    assert c.shape == (
        batch_size,
        hidden_size,
    ), f"Expected cell state shape {(batch_size, hidden_size)}, got {c.shape}"

    # Check value ranges (outputs of sigmoid and tanh should be bounded)
    assert torch.all(h >= -1) and torch.all(
        h <= 1
    ), "Hidden state values outside expected range [-1, 1]"

    print("Single step test passed!")


def test_sequence():
    print("\nTesting sequence forward pass...")

    # Initialize model
    input_size, hidden_size = 10, 20
    lstm = SimpleLSTM(input_size, hidden_size)

    # Test sequence input
    seq_len, batch_size = 15, 32
    x_seq = torch.randn(seq_len, batch_size, input_size)
    outputs, (h_n, c_n) = lstm.forward_sequence(x_seq)

    # Check shapes
    assert outputs.shape == (
        seq_len,
        batch_size,
        hidden_size,
    ), f"Expected outputs shape {(seq_len, batch_size, hidden_size)}, got {outputs.shape}"
    assert h_n.shape == (
        batch_size,
        hidden_size,
    ), f"Expected final hidden state shape {(batch_size, hidden_size)}, got {h_n.shape}"

    # Verify sequence processing
    # Last output should match final hidden state
    assert torch.allclose(
        outputs[-1], h_n
    ), "Last output doesn't match final hidden state"

    print("Sequence test passed!")


def test_state_preservation():
    print("\nTesting state preservation...")

    # Initialize model
    input_size, hidden_size = 10, 20
    lstm = SimpleLSTM(input_size, hidden_size)

    # Generate two inputs
    batch_size = 32
    x1 = torch.randn(batch_size, input_size)
    x2 = torch.randn(batch_size, input_size)

    # Process with and without preserved state
    h1, c1 = lstm(x1)
    h2_with_state, c2_with_state = lstm(x2, state=(h1, c1))
    h2_without_state, c2_without_state = lstm(x2)

    # Verify states are different when using preserved state
    assert not torch.allclose(
        h2_with_state, h2_without_state
    ), "Hidden states should differ when using preserved state"
    assert not torch.allclose(
        c2_with_state, c2_without_state
    ), "Cell states should differ when using preserved state"

    print("State preservation test passed!")


def test_forget_gate_bias():
    print("\nTesting forget gate bias initialization...")

    # Initialize model
    input_size, hidden_size = 10, 20
    lstm = SimpleLSTM(input_size, hidden_size)

    # Check if forget gate bias is initialized to 1
    assert torch.allclose(
        lstm.b_f, torch.ones_like(lstm.b_f)
    ), "Forget gate bias not initialized to 1"

    print("Forget gate bias test passed!")


def benchmark_lstms():
    print("\nRunning LSTM Performance Benchmarks...")

    import gc
    import time

    import torch
    import torch.nn as nn
    from tabulate import tabulate

    # Configuration
    input_size = 10
    hidden_size = 20
    batch_size = 32
    sequence_lengths = [10, 100, 1000]
    num_warmup = 5
    num_trials = 10

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on: {device}")

    # Initialize models
    custom_lstm = SimpleLSTM(input_size, hidden_size).to(device)
    builtin_lstm = nn.LSTM(input_size, hidden_size, batch_first=False).to(device)

    # Set both models to eval mode
    custom_lstm.eval()
    builtin_lstm.eval()

    results = []

    for seq_len in sequence_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")

        # Generate input data
        x = torch.randn(seq_len, batch_size, input_size, device=device)

        # Warmup runs
        print("Warming up...")
        for _ in range(num_warmup):
            with torch.no_grad():
                # Custom LSTM
                custom_lstm.forward_sequence(x)
                # Built-in LSTM
                builtin_lstm(x)

        # Synchronize if using GPU
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark Custom LSTM
        custom_times = []
        custom_memory = []
        for _ in range(num_trials):
            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            start_time = time.perf_counter()
            with torch.no_grad():
                custom_lstm.forward_sequence(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            custom_times.append(time.perf_counter() - start_time)

            if device.type == "cuda":
                custom_memory.append(
                    torch.cuda.max_memory_allocated() / 1024**2
                )  # Convert to MB

        # Benchmark Built-in LSTM
        builtin_times = []
        builtin_memory = []
        for _ in range(num_trials):
            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            start_time = time.perf_counter()
            with torch.no_grad():
                builtin_lstm(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            builtin_times.append(time.perf_counter() - start_time)

            if device.type == "cuda":
                builtin_memory.append(
                    torch.cuda.max_memory_allocated() / 1024**2
                )  # Convert to MB

        # Calculate statistics
        custom_avg_time = sum(custom_times) / len(custom_times) * 1000  # Convert to ms
        builtin_avg_time = (
            sum(builtin_times) / len(builtin_times) * 1000
        )  # Convert to ms
        time_ratio = custom_avg_time / builtin_avg_time

        if device.type == "cuda":
            custom_avg_memory = sum(custom_memory) / len(custom_memory)
            builtin_avg_memory = sum(builtin_memory) / len(builtin_memory)
            memory_ratio = custom_avg_memory / builtin_avg_memory
        else:
            custom_avg_memory = float("nan")
            builtin_avg_memory = float("nan")
            memory_ratio = float("nan")

        results.append(
            [
                seq_len,
                f"{custom_avg_time:.2f}",
                f"{builtin_avg_time:.2f}",
                f"{time_ratio:.2f}x",
                f"{custom_avg_memory:.1f}" if device.type == "cuda" else "N/A",
                f"{builtin_avg_memory:.1f}" if device.type == "cuda" else "N/A",
                f"{memory_ratio:.2f}x" if device.type == "cuda" else "N/A",
            ]
        )

    # Print results table
    headers = [
        "Sequence Length",
        "Custom Time (ms)",
        "Built-in Time (ms)",
        "Time Ratio",
        "Custom Memory (MB)",
        "Built-in Memory (MB)",
        "Memory Ratio",
    ]
    print("\nBenchmark Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))

    # Print scaling analysis
    print("\nScaling Analysis:")
    for i in range(len(sequence_lengths) - 1):
        seq_ratio = sequence_lengths[i + 1] / sequence_lengths[i]
        time_scaling_custom = float(results[i + 1][1]) / float(results[i][1])
        time_scaling_builtin = float(results[i + 1][2]) / float(results[i][2])
        print(f"\nSequence length increase: {seq_ratio}x")
        print(f"Custom LSTM time increase: {time_scaling_custom:.2f}x")
        print(f"Built-in LSTM time increase: {time_scaling_builtin:.2f}x")


if __name__ == "__main__":
    # Run original tests first
    test_single_step()
    test_sequence()
    test_state_preservation()
    test_forget_gate_bias()
    print("\nAll tests passed successfully!")

    # Run benchmarks
    benchmark_lstms()
