import torch
import torch.nn.functional as F

# Define dimensions
input_dim = 3
hidden_dim = 4
output_dim = 2

# Input vector
x = torch.tensor([1.0, 2.0, 3.0])  # Shape: (3,)
print("Input vector (x):")
print(x)

# Layer 1: Initialize weights and biases
W1 = torch.randn(hidden_dim, input_dim)  # Shape: (4, 3)
b1 = torch.randn(hidden_dim)  # Shape: (4,)

# Compute z1 (linear transformation) and a1 (ReLU activation)
z1 = torch.matmul(W1, x) + b1
a1 = F.relu(z1)

print("\nLayer 1:")
print("Weights (W1):")
print(W1)
print("Bias (b1):")
print(b1)
print("Linear output (z1):")
print(z1)
print("Activation output (a1):")
print(a1)

# Layer 2: Initialize weights and biases
W2 = torch.randn(output_dim, hidden_dim)  # Shape: (2, 4)
b2 = torch.randn(output_dim)  # Shape: (2,)

# Compute z2 (linear transformation) and a2 (ReLU activation)
z2 = torch.matmul(W2, a1) + b2
a2 = F.relu(z2)

print("\nLayer 2:")
print("Weights (W2):")
print(W2)
print("Bias (b2):")
print(b2)
print("Linear output (z2):")
print(z2)
print("Activation output (a2):")
print(a2)