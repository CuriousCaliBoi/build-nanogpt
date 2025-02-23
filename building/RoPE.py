import torch

def get_rotary_embeddings(seq_len, head_dim, base=10000):
    """
    Compute rotary embeddings (cosine and sine) for a given sequence length and head dimension.

    Args:
        seq_len (int): Maximum sequence length.
        head_dim (int): Dimensionality of each attention head (must be even).
        base (float): The base used in the inverse frequency computation.

    Returns:
        cos (torch.Tensor): Tensor of shape (seq_len, head_dim) with cosine values.
        sin (torch.Tensor): Tensor of shape (seq_len, head_dim) with sine values.
    """
    assert head_dim % 2 == 0, "head_dim must be even."
    half_dim = head_dim // 2
    # Compute inverse frequencies for each pair of dimensions.
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    # Create a positions tensor (shape: [seq_len])
    positions = torch.arange(seq_len, dtype=torch.float32)
    # Compute the angles using an outer product: shape (seq_len, half_dim)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    # Duplicate the angles for each half to form full head_dim: shape (seq_len, head_dim)
    cos = torch.cos(torch.cat([angles, angles], dim=-1))
    sin = torch.sin(torch.cat([angles, angles], dim=-1))
    
    return cos, sin

def rotate_half(x):
    """
    Splits the last dimension into two halves and rotates them.
    For each pair (a, b), returns (-b, a).

    Args:
        x (torch.Tensor): Tensor of shape (..., head_dim)
        
    Returns:
        torch.Tensor: Tensor of the same shape as x.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """
    Applies rotary positional embeddings to tensor x.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, seq_len, num_heads, head_dim).
        cos (torch.Tensor): Cosine embeddings of shape (seq_len, head_dim).
        sin (torch.Tensor): Sine embeddings of shape (seq_len, head_dim).

    Returns:
        torch.Tensor: The tensor x after applying rotary embeddings.
    """
    # Reshape cos and sin to be broadcastable: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    return x * cos + rotate_half(x) * sin

if __name__ == "__main__":
    # Example dimensions
    batch = 2
    seq_len = 128
    num_heads = 12
    head_dim = 64  # Must be even

    # Create a dummy query tensor
    x = torch.randn(batch, seq_len, num_heads, head_dim)

    # Precompute rotary embeddings for the given sequence length and head dimension
    cos, sin = get_rotary_embeddings(seq_len, head_dim)

    # Apply rotary positional embeddings to x
    x_rot = apply_rotary_pos_emb(x, cos, sin)

    # Print shapes to verify everything worked correctly
    print("Original x shape:", x.shape)
    print("Rotated x shape:", x_rot.shape)