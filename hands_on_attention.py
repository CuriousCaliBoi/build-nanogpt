import torch
import math

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    print("Initial query shape:", query.shape)
    print("Initial key shape:", key.shape)
    print("Initial value shape:", value.shape)
    print("Scale factor:", scale_factor)
    print("Initial attention bias:", attn_bias)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        print("Causal mask applied to attention bias:", attn_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
        print("Attention bias after applying attn_mask:", attn_bias)

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
        print("Key shape after GQA:", key.shape)
        print("Value shape after GQA:", value.shape)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    print("Attention weights before bias and softmax:", attn_weight)

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    print("Attention weights after softmax:", attn_weight)

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    print("Attention weights after dropout:", attn_weight)

    output = attn_weight @ value
    print("Output shape:", output.shape)
    return output

# Example usage with random input
B, T, C = 4, 8, 64  # Batch size, sequence length, embedding dimension
query = torch.randn(B, T, C)
key = torch.randn(B, T, C)
value = torch.randn(B, T, C)

# Call the function with random inputs
output = scaled_dot_product_attention(query, key, value, is_causal=True)
print("Final output:", output)