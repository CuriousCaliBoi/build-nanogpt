import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)

def visualize_layer_output(input_data, output_data, title):
    plt.figure(figsize=(12, 4))
    
    # Input visualization
    plt.subplot(121)
    sns.heatmap(input_data.detach().numpy(), cmap='viridis', annot=True, fmt='.2f')
    plt.title('Input Data')
    
    # Output visualization
    plt.subplot(122)
    sns.heatmap(output_data.detach().numpy(), cmap='viridis', annot=True, fmt='.2f')
    plt.title(f'After {title}')
    
    plt.tight_layout()
    plt.show()

# 1. Linear Layer
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)  # 3 input features, 2 output features
    
    def forward(self, x):
        return self.linear(x)

# 2. ReLU Activation
class SimpleReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x)

# 3. LayerNorm
class SimpleLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(3)  # normalize over 3 features
    
    def forward(self, x):
        return self.ln(x)

# Create sample data
x = torch.randn(4, 3)  # 4 samples, 3 features

# Test and visualize each layer
# Linear Layer
linear_model = SimpleLinear()
linear_out = linear_model(x)
visualize_layer_output(x, linear_out, "Linear Layer")

# ReLU
relu_model = SimpleReLU()
relu_out = relu_model(x)
visualize_layer_output(x, relu_out, "ReLU")

# LayerNorm
ln_model = SimpleLayerNorm()
ln_out = ln_model(x)
visualize_layer_output(x, ln_out, "LayerNorm")

# Print layer parameters
print("\nLayer Parameters:")
print("Linear Layer weights:\n", linear_model.linear.weight.data)
print("Linear Layer bias:\n", linear_model.linear.bias.data)
print("\nLayerNorm parameters:")
print("gamma (weight):", ln_model.ln.weight.data)
print("beta (bias):", ln_model.ln.bias.data)