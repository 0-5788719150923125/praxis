import torch

# Input and target output
x = torch.tensor([1.0])  # Single input
y_true = torch.tensor([0.0])  # Target output

# Initialize weights and biases
w1 = torch.tensor([0.5], requires_grad=True)  # Weight for hidden layer
b1 = torch.tensor([0.0], requires_grad=True)  # Bias for hidden layer
w2 = torch.tensor([0.5], requires_grad=True)  # Weight for output layer
b2 = torch.tensor([0.0], requires_grad=True)  # Bias for output layer

# Hidden layer computation
z1 = w1 * x + b1
h = 1 / (1 + torch.exp(-z1))  # Sigmoid activation

# Output layer computation
y_pred = w2 * h + b2

# Reset gradients
if w1.grad is not None:
    w1.grad.zero_()
    b1.grad.zero_()
    w2.grad.zero_()
    b2.grad.zero_()

# Perform forward pass again
z1 = w1 * x + b1
h = 1 / (1 + torch.exp(-z1))
y_pred = w2 * h + b2

# Compute loss
loss = 0.5 * (y_pred - y_true) ** 2

# Perform backward pass
loss.backward()

# Display gradients
print(f"Gradients computed by PyTorch:")
print(f"w1.grad = {w1.grad.item()}")
print(f"b1.grad = {b1.grad.item()}")
print(f"w2.grad = {w2.grad.item()}")
print(f"b2.grad = {b2.grad.item()}")
