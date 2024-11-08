import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Generate toy dataset
X = torch.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1]).unsqueeze(1)  # Target is sum of features

# Initialize model, loss function and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_X = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    predictions = model(test_X)
    print("\nTest Predictions:")
    print(f"Input: {test_X}")
    print(f"Predicted: {predictions.squeeze()}")
    print(f"Expected: {test_X[:, 0] + test_X[:, 1]}")
