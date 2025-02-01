import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn


class FPROPTrainer:
    def __init__(self, model, learning_rate=0.01, device="cpu", num_probes=10):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.num_probes = num_probes

    def generate_synthetic_data(self, num_samples, input_dim):
        X = torch.randn(num_samples, input_dim, device=self.device)
        # Create target = sum of inputs plus noise
        y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(
            num_samples, 1, device=self.device
        )
        return X, y

    def print_tensor_stats(self, tensor, name):
        """Print useful statistics about a tensor"""
        if tensor is None:
            print(f"{name} is None!")
            return
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Mean: {tensor.mean().item():.6f}")
        print(f"  Std: {tensor.std().item():.6f}")
        print(f"  Min: {tensor.min().item():.6f}")
        print(f"  Max: {tensor.max().item():.6f}")
        print(f"  Num zeros: {(tensor == 0).sum().item()}")
        if torch.isnan(tensor).any():
            print("  WARNING: Contains NaN values!")
        if torch.isinf(tensor).any():
            print("  WARNING: Contains Inf values!")

    def fprop_step(self, x, y, epoch=None, batch_idx=None):
        batch_size = x.size(0)

        # Initialize accumulated updates tensor
        accumulated_update = torch.zeros_like(
            x
        )  # This will store our net update direction

        # Multiple probes for better gradient estimation
        for _ in range(self.num_probes):
            with fwAD.dual_level():
                # Create dual input
                tangent = torch.randn_like(x)
                tangent = tangent / torch.norm(tangent)
                dual_input = fwAD.make_dual(x, tangent)

                # Forward pass
                dual_output = self.model(dual_input)
                primal_output, jvp = fwAD.unpack_dual(dual_output)

                # Compute loss and gradient
                loss = self.criterion(primal_output, y)
                loss_grad = 2 * (primal_output - y) / batch_size

                # Compute update using JVP
                update = jvp * loss_grad
                accumulated_update.add_(update / self.num_probes)

        # Update parameters based on accumulated update
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data.sub_(self.learning_rate * accumulated_update.mean())

        return loss.item()

    def train(self, num_epochs, batch_size, input_dim):
        X, y = self.generate_synthetic_data(1000, input_dim)

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = len(X) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]

                loss = self.fprop_step(batch_X, batch_y, epoch, i)
                total_loss += loss

            avg_loss = total_loss / num_batches
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.8f}")


def main():
    torch.manual_seed(42)  # For reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 3

    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    trainer = FPROPTrainer(model, learning_rate=0.01, device=device, num_probes=10)

    trainer.train(num_epochs=1000, batch_size=32, input_dim=input_dim)


if __name__ == "__main__":
    main()
