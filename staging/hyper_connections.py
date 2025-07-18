import torch
import torch.nn as nn


class HyperConnections(nn.Module):
    def __init__(self, transformer_layer, expansion_rate=4, dim=512):
        super().__init__()
        self.transformer = transformer_layer
        self.n = expansion_rate
        self.dim = dim

        # Learnable scalar weights
        # Beta weights for transformer output
        self.beta = nn.Parameter(torch.ones(self.n))
        # Alpha weights for residual connections
        self.alpha = nn.Parameter(torch.ones(self.n, self.n))

        # Initialize weights
        with torch.no_grad():
            self.beta.fill_(1.0 / self.n)
            self.alpha.fill_(1.0 / self.n)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: List of n tensors, each of shape [batch, seq_len, dim]
        Returns:
            List of n tensors, each of shape [batch, seq_len, dim]
        """
        # Compute weighted sum of hidden states for transformer input
        combined_input = sum(h * self.alpha[0][i] for i, h in enumerate(hidden_states))

        # Pass through transformer layer once
        transformer_output = self.transformer(combined_input)

        # Compute new hidden states
        new_hidden_states = []
        for i in range(self.n):
            # Weighted sum of previous hidden states
            residual = sum(h * self.alpha[i][j] for j, h in enumerate(hidden_states))
            # Add weighted transformer output
            new_state = transformer_output * self.beta[i] + residual
            new_hidden_states.append(new_state)

        return new_hidden_states


# Example usage
if __name__ == "__main__":
    batch_size, seq_len, dim = 32, 128, 512
    n = 4  # expansion rate

    # Initialize n hidden states
    hidden_states = [torch.randn(batch_size, seq_len, dim) for _ in range(n)]

    # Create model (assuming transformer_layer is defined)
    class DummyTransformer(nn.Module):
        def forward(self, x):
            return x

    model = HyperConnections(DummyTransformer(), expansion_rate=n, dim=dim)

    # Forward pass
    new_hidden_states = model(hidden_states)
    print(f"Number of hidden states: {len(new_hidden_states)}")
    print(f"Shape of each hidden state: {new_hidden_states[0].shape}")
