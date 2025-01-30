from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_ctx: int = 1024
    n_vocab: int = 50257

#______________________________________________________________________
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

# yer seems legit
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        # but like I get whats happening with all these layer initialization
        #I can go through the indiivdual layers later
        # dont have the mlp clas or the causal self attention class yet
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# GPT model class that inherits from PyTorch's nn.Module base class
class GPT(nn.Module):
    def __init__(self, config):
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Store the config object as an instance variable for later use
        self.config = config
        
        # Create a ModuleDict to store the main components of the transformer
        self.transformer = nn.ModuleDict(dict(
            # wte: Word Token Embeddings - converts token IDs to vectors
            # Takes vocab size and embedding dimension from config
            wte = nn.Embedding(config.n_vocab, config.n_embd),
            
            # h: Stack of Transformer blocks
            # Creates n_layer identical transformer blocks using list comprehension
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # ln_f: Final layer normalization
            # Normalizes the output embeddings before final projection
            ln_f = nn.LayerNorm(config.n_embd),
        ))
