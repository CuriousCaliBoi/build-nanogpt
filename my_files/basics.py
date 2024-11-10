import torch
import torch.nn as nn

# A simple linear layer
linear = nn.Linear(3, 2)  # Input size 3, output size 2

# Random input
x = torch.randn(1, 3)  # Batch of 1, 3 features
print("Input:", x)

# Output after linear transformation
output = linear(x)
print("Output:", output)