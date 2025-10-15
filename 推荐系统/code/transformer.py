import torch
import torch.nn as nn

# 注意力模块
def attention(Q, K, V, mask = None):
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nums_heads):
        super().__init__()
        assert d_model % nums_heads == 0
        self.d_k = d_model // nums_heads
        self.nums_head = nums_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, L, D = Q.size()
        Q = self.W_q(Q).view(B, L, self.nums_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, L, self.nums_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, L, self.nums_heads, self.d_k).transpose(1, 2)
        out = attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
