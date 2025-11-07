import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 2. 缩放点积注意力
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output, attn


# 3. 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        output, attn = scaled_dot_product_attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output), attn