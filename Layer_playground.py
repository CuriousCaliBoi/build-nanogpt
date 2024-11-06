import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's play with each layer type separately:

# 1. ModuleDict Playground
class ModuleDictDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_dict = nn.ModuleDict({
            'linear1': nn.Linear(10, 5),
            'linear2': nn.Linear(5, 2),
            'embedding': nn.Embedding(100, 10)
        })
    
    def forward(self, x, mode='linear1'):
        return self.layer_dict[mode](x)

# Try it out:
dict_model = ModuleDictDemo()
# For linear layers
x_linear = torch.randn(3, 10)  # 3 samples, 10 features
print("Linear1 output:", dict_model(x_linear, 'linear1').shape)  # [3, 5]

# For embedding
x_emb = torch.tensor([[1,2,3], [4,5,6]])  # 2 sequences of 3 tokens
print("Embedding output:", dict_model(x_emb, 'embedding').shape)  # [2, 3, 10]

# 2. ModuleList Playground
class ModuleListDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 8),
            nn.Linear(8, 6),
            nn.Linear(6, 4)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# Try it out:
list_model = ModuleListDemo()
x = torch.randn(2, 10)
print("Sequential output:", list_model(x).shape)  # [2, 4]

# 3. LayerNorm Playground
class LayerNormDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(10)  # normalize over last dimension
    
    def forward(self, x):
        # Print stats before normalization
        print("Before LN - mean:", x.mean().item(), "std:", x.std().item())
        x = self.ln(x)
        p