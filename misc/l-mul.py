import torch
from torch.autograd import Function


class LmulFunction(Function):
    @staticmethod
    def forward(ctx, x, y, mantissa_bits):
        # Save tensors for backward
        ctx.save_for_backward(x, y)
        ctx.mantissa_bits = mantissa_bits

        # Implement the lmul forward pass
        result = lmul_approximation(x, y, mantissa_bits)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        mantissa_bits = ctx.mantissa_bits

        # Approximate gradients using lmul
        grad_x = lmul_approximation(grad_output, y, mantissa_bits)
        grad_y = lmul_approximation(grad_output, x, mantissa_bits)

        return grad_x, grad_y, None  # None for mantissa_bits


def lmul(x, y, mantissa_bits=10):
    return LmulFunction.apply(x, y, mantissa_bits)


def lmul_approximation(x, y, mantissa_bits):
    x = x.float()
    y = y.float()

    # Handle zero inputs
    x_zero_mask = x == 0
    y_zero_mask = y == 0

    # Extract mantissa and exponent
    x_mantissa, x_exponent = torch.frexp(x)
    y_mantissa, y_exponent = torch.frexp(y)

    # Replace mantissa and exponent for zero values
    x_mantissa = x_mantissa.masked_fill(x_zero_mask, 0)
    x_exponent = x_exponent.masked_fill(x_zero_mask, 0)
    y_mantissa = y_mantissa.masked_fill(y_zero_mask, 0)
    y_exponent = y_exponent.masked_fill(y_zero_mask, 0)

    # Quantize mantissas
    quantization_level = 2**mantissa_bits
    x_mantissa_q = torch.round(x_mantissa * quantization_level) / quantization_level
    y_mantissa_q = torch.round(y_mantissa * quantization_level) / quantization_level

    # Define l(m)
    if mantissa_bits <= 3:
        l_m = mantissa_bits
    elif mantissa_bits == 4:
        l_m = 3
    else:
        l_m = 4

    offset = 2 ** (-l_m)

    # Approximate multiplication
    lmul_mantissa = x_mantissa_q + y_mantissa_q + offset
    lmul_exponent = x_exponent + y_exponent
    result = torch.ldexp(lmul_mantissa, lmul_exponent)

    # Handle zero inputs
    result = result.masked_fill(x_zero_mask | y_zero_mask, 0)

    return result


import torch.nn as nn


class LmulLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mantissa_bits=10):
        super(LmulLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mantissa_bits = mantissa_bits
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = lmul_matmul(input, self.weight.t(), self.mantissa_bits)
        if self.bias is not None:
            output += self.bias
        return output


def lmul_matmul(input, weight, mantissa_bits=10):
    return torch.sum(
        lmul(input.unsqueeze(2), weight.unsqueeze(0), mantissa_bits), dim=1
    )


class LmulMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mantissa_bits=10):
        super(LmulMLP, self).__init__()
        self.layer1 = LmulLinear(input_size, hidden_size, mantissa_bits=mantissa_bits)
        self.relu = nn.ReLU()
        self.layer2 = LmulLinear(hidden_size, output_size, mantissa_bits=mantissa_bits)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


import torch.optim as optim

# Generate a simple dataset
torch.manual_seed(0)
num_samples = 1000
X = torch.randn(num_samples, 2)

# Normalize input data
X = X / X.abs().max()

Y = (X[:, 0] * X[:, 1] > 0).long()  # Label is 1 if x and y have the same sign

# Split into training and test sets
train_X, test_X = X[:800], X[800:]
train_Y, test_Y = Y[:800], Y[800:]

# Initialize the model
model = LmulMLP(input_size=2, hidden_size=64, output_size=2, mantissa_bits=10)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_X)
    loss = criterion(outputs, train_Y)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == train_Y).sum().item() / train_Y.size(0)
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%"
        )

# Test the model
with torch.no_grad():
    test_outputs = model(test_X)
    _, predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (predicted == test_Y).sum().item() / test_Y.size(0)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
