import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward  import PositionWiseFeedForward
from PositionalEncoding import PositionalEncoding
import math

# 6. Decoder层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

# 8. 完整Decoder
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
        return x
