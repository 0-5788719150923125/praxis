import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, leak_factor=0.9, reset_value=0.0):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.reset_value = reset_value
        self.membrane_potential = None

    def reset_membrane(self):
        self.membrane_potential = None

    def forward(self, input_tensor):
        # Initialize membrane potential with correct shape if None
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(input_tensor)

        # Ensure membrane potential has same shape as input
        if self.membrane_potential.shape != input_tensor.shape:
            self.membrane_potential = torch.zeros_like(input_tensor)

        self.membrane_potential = self.membrane_potential * self.leak_factor
        self.membrane_potential = self.membrane_potential + input_tensor

        spikes = torch.zeros_like(self.membrane_potential)
        spikes[self.membrane_potential >= self.threshold] = 1.0
        self.membrane_potential[spikes == 1.0] = self.reset_value

        return spikes


class EnhancedSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnhancedSNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Two fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Two LIF layers
        self.lif1 = LIFNeuron(threshold=0.8)
        self.lif2 = LIFNeuron(threshold=0.5)

        # Readout layer
        self.readout = nn.Linear(output_size, 2)

    def reset_neurons(self):
        self.lif1.reset_membrane()
        self.lif2.reset_membrane()

    def forward(self, x, num_steps):
        batch_size = x.size(0)
        spike_history = []

        # Reset neuron states at the start of each sequence
        self.reset_neurons()

        for t in range(num_steps):
            # First layer
            current1 = self.fc1(x)
            spikes1 = self.lif1(current1)

            # Second layer
            current2 = self.fc2(spikes1)
            spikes2 = self.lif2(current2)

            spike_history.append(spikes2)

        # Stack spikes and average over time steps
        spike_tensor = torch.stack(
            spike_history, dim=1
        )  # [batch_size, num_steps, output_size]
        spike_rate = torch.mean(spike_tensor, dim=1)  # [batch_size, output_size]

        # Convert spike rates to classification
        output = self.readout(spike_rate)  # [batch_size, 2]
        return output, spike_tensor


class VibrationDataset(Dataset):
    def __init__(self, num_samples, input_size, sequence_length):
        self.num_samples = num_samples
        self.input_size = input_size
        self.sequence_length = sequence_length

        # Generate synthetic data
        self.data = []
        self.labels = []

        for i in range(num_samples):
            if np.random.random() > 0.5:
                # Normal pattern
                pattern = self._generate_normal_pattern()
                self.labels.append(0)
            else:
                # Anomaly pattern
                pattern = self._generate_anomaly_pattern()
                self.labels.append(1)

            self.data.append(pattern)

        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def _generate_normal_pattern(self):
        # Generate regular vibration pattern with small random variations
        base_frequency = 0.1
        time = np.linspace(0, 10, self.input_size)
        pattern = np.sin(2 * np.pi * base_frequency * time)
        noise = np.random.normal(0, 0.1, self.input_size)
        return pattern + noise

    def _generate_anomaly_pattern(self):
        # Generate irregular vibration pattern with sudden spikes
        pattern = self._generate_normal_pattern()
        # Add random spikes
        spike_positions = np.random.choice(self.input_size, 3, replace=False)
        pattern[spike_positions] += np.random.uniform(0.5, 1.5, 3)
        return pattern

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_network():
    # Parameters
    input_size = 100
    hidden_size = 50
    output_size = 20
    num_steps = 50
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # Create network and dataset
    model = EnhancedSNN(input_size, hidden_size, output_size)
    train_dataset = VibrationDataset(1000, input_size, num_steps)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        epoch_losses = []
        all_preds = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            outputs, spike_tensor = model(data, num_steps)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Record metrics
            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

        # Calculate epoch metrics
        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return model, train_losses, train_accuracies


def visualize_results(model, losses, accuracies):
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


def demonstrate_prediction(model, num_steps=50):
    # Generate one normal and one anomaly sample
    dataset = VibrationDataset(2, 100, num_steps)
    normal_data = dataset.data[0].unsqueeze(0)
    anomaly_data = dataset.data[1].unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        normal_output, normal_spikes = model(normal_data, num_steps)
        anomaly_output, anomaly_spikes = model(anomaly_data, num_steps)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot input signals
    plt.subplot(2, 2, 1)
    plt.plot(normal_data[0].numpy())
    plt.title("Normal Vibration Pattern")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 2, 2)
    plt.plot(anomaly_data[0].numpy())
    plt.title("Anomaly Vibration Pattern")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Plot spike patterns
    plt.subplot(2, 2, 3)
    plt.imshow(normal_spikes[0].T.numpy(), aspect="auto", cmap="Blues")
    plt.title("Normal Pattern Spike Activity")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Index")

    plt.subplot(2, 2, 4)
    plt.imshow(anomaly_spikes[0].T.numpy(), aspect="auto", cmap="Blues")
    plt.title("Anomaly Pattern Spike Activity")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train the network
    model, losses, accuracies = train_network()

    # Visualize training results
    visualize_results(model, losses, accuracies)

    # Demonstrate predictions
    demonstrate_prediction(model)
