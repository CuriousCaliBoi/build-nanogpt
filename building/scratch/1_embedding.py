import torch
import torch.nn as nn

class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length):
        super().__init__()
        # Token embeddings (similar to wte in GPT-2)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Position embeddings (similar to wpe in GPT-2)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Optionally, initialize the embeddings as in GPT-2 with a normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Args:
            input_ids (LongTensor): Tensor of shape (batch_size, sequence_length)
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim) representing
            the sum of token and positional embeddings.
        """
        # Compute token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Create position ids for each token
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        # Compute position embeddings
        position_embeds = self.position_embedding(position_ids)
        
        # Sum the token and position embeddings
        embeddings = token_embeds + position_embeds
        return embeddings
