import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # def forward(self, x):
    #     B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    #     # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    #     # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    #     # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    #     qkv = self.c_attn(x)
    #     q, k, v = qkv.split(self.n_embd, dim=2)
    #     k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
    #     y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    #     # output projection
    #     y = self.c_proj(y)
    #     return y
    def forward(self, x):
        # Input shape: (batch_size, seq_len, n_embd)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        print(f"Input x shape: {x.shape}")  # e.g. (4, 8, 768)
        
        # Project input to get query, key, value vectors concatenated
        qkv = self.c_attn(x)  # Shape: (B, T, 3*C)
        print(f"After c_attn projection shape: {qkv.shape}")  # e.g. (4, 8, 2304)
       
        # Split into query, key, value
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each shape: (B, T, C)
        print(f"After splitting - q,k,v shapes: {q.shape}")  # e.g. (4, 8, 768)
        
        # Reshape q,k,v for multi-head attention
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        print(f"After reshaping - q,k,v shapes: {q.shape}")  # e.g. (4, 12, 8, 64)
        print("Key tensor:")
        print(k)
        print("Query tensor:")
        print(q) 
        print("Value tensor:")
        print(v)

        # Apply scaled dot-product attention (using flash attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"After attention shape: {y.shape}")  # e.g. (4, 12, 8, 64)
        print("Attention tensor: I want to see the different heads before we reshape back")
        print(y)
        # Reshape back to original dimensions
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        print(f"After reshaping back: {y.shape}")  # e.g. (4, 8, 768)
        
        # Final projection
        y = self.c_proj(y)
        print(f"Final output shape: {y.shape}")  # e.g. (4, 8, 768)
        
        return y
        
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Create a random tensor with batch size 4, sequence length 8, and embedding dim 768
B, T, C = 4, 8, 768
x = torch.randn(B, T, C)

# Create a simple config object with required attributes
class SimpleConfig:
    def __init__(self):
        self.n_head = 12  # Number of attention heads, typical for GPT-style models
        self.n_embd = C   # Embedding dimension

config = SimpleConfig()

# Instantiate CausalSelfAttention
causal_attn = CausalSelfAttention(config)

# Forward pass through the attention layer
output = causal_attn(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
