import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, leak_factor=0.9, reset_value=0.0):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.reset_value = reset_value
        self.membrane_potential = None

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)

        # Initialize membrane potential if None
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(input_tensor)

        # Apply leak
        self.membrane_potential = self.membrane_potential * self.leak_factor

        # Add input current to membrane potential
        self.membrane_potential = self.membrane_potential + input_tensor

        # Generate spikes where membrane potential exceeds threshold
        spikes = torch.zeros_like(self.membrane_potential)
        spikes[self.membrane_potential >= self.threshold] = 1.0

        # Reset membrane potential where spikes occurred
        self.membrane_potential[spikes == 1.0] = self.reset_value

        return spikes


class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SpikingNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Single fully connected layer
        self.fc = nn.Linear(input_size, output_size)

        # LIF neuron layer
        self.lif = LIFNeuron()

    def forward(self, x, num_steps):
        batch_size = x.size(0)
        spikes_record = []

        # Simulate network for num_steps time steps
        for t in range(num_steps):
            # Pass input through fully connected layer
            current = self.fc(x)

            # Pass current through LIF neurons
            spikes = self.lif(current)
            spikes_record.append(spikes)

        # Stack recorded spikes along time dimension
        return torch.stack(spikes_record, dim=1)


# Example usage
def main():
    # Create network
    input_size = 10
    output_size = 5
    num_steps = 100
    batch_size = 32

    snn = SpikingNeuralNetwork(input_size, output_size)

    # Create random input spikes
    input_spikes = torch.rand(batch_size, input_size) > 0.5
    input_spikes = input_spikes.float()

    # Run simulation
    output_spikes = snn(input_spikes, num_steps)

    # output_spikes shape: [batch_size, num_steps, output_size]
    print(f"Output shape: {output_spikes.shape}")


if __name__ == "__main__":
    main()
