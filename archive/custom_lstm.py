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


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

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

    # Run all tests
    test_single_step()
    test_sequence()
    test_state_preservation()
    test_forget_gate_bias()
    print("\nAll tests passed successfully!")
