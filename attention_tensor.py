import torch
import torch.nn as nn
import torch.nn.functional as F

# Set parameters for the demo
B = 2           # Batch size (number of sequences)
T = 4           # Sequence length (number of tokens per sequence)
C = 8           # Embedding dimension (for demonstration; typical value is 768)
n_head = 2      # Number of attention heads
head_size = C // n_head  # Size per head (here, 8 / 2 = 4)

# Create a fake input tensor x of shape (B, T, C)
x = torch.arange(B * T * C, dtype=torch.float32).view(B, T, C)
print("Input x shape:", x.shape)
print(x)

# Create a fake linear layer that simulates c_attn
# It projects from dimension C to 3 * C (to obtain q, k, v concatenated)
c_attn = nn.Linear(C, 3 * C)

# Apply the linear layer to x. The output shape will be (B, T, 3*C)
qkv = c_attn(x)
print("\nAfter c_attn, qkv shape:", qkv.shape)

# Split qkv into query, key, and value tensors along the last dimension.
# Each will have shape (B, T, C)
q, k, v = qkv.split(C, dim=2)
print("Query shape:", q.shape)
print("Key shape:", k.shape)
print("Value shape:", v.shape)

# Reshape each of q, k, and v to prepare for multi-head attention.
# We first reshape from (B, T, C) to (B, T, n_head, head_size)
# Then, we transpose dimensions to get (B, n_head, T, head_size)
q = q.view(B, T, n_head, head_size).transpose(1, 2)
k = k.view(B, T, n_head, head_size).transpose(1, 2)
v = v.view(B, T, n_head, head_size).transpose(1, 2)

print("\nAfter reshaping and transposing:")
print("Query shape:", q.shape)  # (B, n_head, T, head_size)
print("Key shape:", k.shape)    # (B, n_head, T, head_size)
print("Value shape:", v.shape)  # (B, n_head, T, head_size)

# --- Scaled Dot-Product Attention (without using flash attention) ---
# Compute the dot products between queries and keys:
# We first transpose k on the last two dimensions to perform matrix multiplication.
scale = head_size ** 0.5  # scaling factor, typically sqrt(head_size)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
print("\nAttention scores shape:", attn_scores.shape)

# Apply softmax to obtain attention weights (probabilities) along the last dimension.
attn_weights = F.softmax(attn_scores, dim=-1)
print("Attention weights shape:", attn_weights.shape)

# Multiply the attention weights with the values.
attn_output = torch.matmul(attn_weights, v)
print("Attention output (per head) shape:", attn_output.shape)

# --- Reassemble the multi-head outputs ---
# Currently, attn_output has shape (B, n_head, T, head_size)
# We transpose back to (B, T, n_head, head_size)
# Then, we reshape (flatten) the last two dimensions to get (B, T, C)
output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
print("Final output shape:", output.shape)