import torch
import torch.nn as nn
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Generate dummy data
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 3: Define the SVM model
class SVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# class PraxiSVM(nn.Module):
#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.n_dim = config.n_dim
#         self.key = nn.Linear(self.n_dim, config.n_layer)
#         self.temperature = 0.9

#     def forward(self, hidden_states, labels=None):
#         # Compute the router logits
#         logits = self.key(hidden_states[:, -1])  # Use the last token for routing

#         # Add Gumbel noise to the logits
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#         noisy_logits = (logits + gumbel_noise) / self.temperature

#         # Apply softmax to obtain the permutation
#         permutation = F.softmax(noisy_logits, dim=-1)

#         # Get the indices of the experts in the permutation order
#         expert_order = torch.argsort(permutation, dim=-1, descending=True)

#         # Compute the hinge loss if labels are provided
#         hinge_loss = 0
#         if labels is not None:
#             hinge_loss = nn.HingeEmbeddingLoss()(logits.squeeze(), labels.float())

#         return expert_order, hinge_loss


# class PraxiSVM(nn.Module):
#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.n_dim = config.n_dim
#         self.hidden_size = config.n_dim // 2
#         self.temporal = nn.GRU(self.n_dim, self.hidden_size, batch_first=True)
#         self.out = nn.Linear(self.hidden_size, config.n_layer)
#         self.temperature = 0.9

#     def forward(self, hidden_states, labels=None):
#         # Pass the hidden states through the GRU
#         replay_output, _ = self.temporal(hidden_states)
#         logits = self.out(replay_output[:, -1])

#         # Add Gumbel noise to the logits
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#         noisy_logits = (logits + gumbel_noise) / self.temperature

#         # Apply softmax to obtain the permutation
#         permutation = F.softmax(noisy_logits, dim=-1)

#         # Get the indices of the experts in the permutation order
#         expert_order = torch.argsort(permutation, dim=-1, descending=True)

#         # Compute the hinge loss if labels are provided
#         hinge_loss = 0
#         if labels is not None:
#             hinge_loss = nn.HingeEmbeddingLoss()(logits.squeeze(), labels.float())

#         return expert_order, hinge_loss


if __name__ == "__main__":
    model = SVM(input_dim=2)

    # Step 4: Define the loss function and optimizer
    criterion = nn.HingeEmbeddingLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Step 5: Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train.float())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Step 6: Evaluate the model on the test set
    with torch.no_grad():
        outputs = model(X_test)
        predicted = torch.sign(outputs).squeeze().long()
        accuracy = accuracy_score(y_test, predicted)
        print(f"Test Accuracy: {accuracy:.4f}")
