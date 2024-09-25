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
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


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
