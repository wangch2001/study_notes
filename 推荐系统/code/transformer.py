import torch
import torch.nn as nn

# -------------------
# Attention 基础模块
# -------------------
def attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, L, D = Q.size()
        Q = self.W_q(Q).view(B, L, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(K).view(B, K.size(1), self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(V).view(B, V.size(1), self.num_heads, self.d_k).transpose(1,2)
        out = attention(Q, K, V, mask)
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.W_o(out)

# -------------------
# Feed Forward
# -------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# -------------------
# Encoder
# -------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# -------------------
# Decoder
# -------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)

        # Encoder-Decoder Attention
        enc_dec_out = self.enc_dec_attn(x, enc_out, enc_out, memory_mask)
        x = self.norm2(x + enc_dec_out)

        # Feed Forward
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x

# -------------------
# Seq2Seq Transformer
# -------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, num_heads=8, d_ff=512, num_layers=2, max_len=512):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model, num_heads, d_ff, num_layers, max_len)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, num_heads, d_ff, num_layers, max_len)
        self.out_proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, memory_mask)
        return self.out_proj(dec_out)


# 假设：源语言字典大小 1000，目标语言字典大小 1200
model = Transformer(src_vocab=1000, tgt_vocab=1200)

src = torch.randint(0, 1000, (32, 20))  # batch=32, 源序列长=20
tgt = torch.randint(0, 1200, (32, 15))  # batch=32, 目标序列长=15

out = model(src, tgt)  # [32, 15, 1200] → 每个位置输出一个目标词分布
print(out.shape)


# 输出
torch.Size([32, 15, 1200])