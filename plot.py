import plotext as plt
import time
import random


def simulate_training():
    """Simulate training process and yield loss values."""
    for epoch in range(100):
        loss = 1 / (epoch + 1) + random.uniform(0, 0.1)
        yield epoch, loss


def plot_loss_curve():
    epochs = []
    losses = []

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    for epoch, loss in simulate_training():
        epochs.append(epoch)
        losses.append(loss)

        plt.clf()
        plt.plot(epochs, losses)
        plt.show()

        time.sleep(0.1)  # Simulate some processing time


if __name__ == "__main__":
    plot_loss_curve()
