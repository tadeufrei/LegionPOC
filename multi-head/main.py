import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values):
        batch_size = queries.size(0)

        Q = self.split_heads(self.W_q(queries), batch_size)
        K = self.split_heads(self.W_k(keys), batch_size)
        V = self.split_heads(self.W_v(values), batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        scaled_attention = torch.matmul(attention_weights, V)

        # Concatenate heads
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.fc_out(concat_attention)
        return output

# Example usage
queries = torch.rand(3, 10, 512)  # (batch_size, seq_len, d_model)
keys = torch.rand(3, 10, 512)
values = torch.rand(3, 10, 512)

multi_head_attention = MultiHeadAttention(d_model=512, num_heads=8)
output = multi_head_attention(queries, keys, values)
print("Multi-Head Attention Output Shape:", output.shape)