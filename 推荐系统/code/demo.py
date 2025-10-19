import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        多头自注意力机制（Multi-Head Self-Attention）
        参数：
            embed_dim: 每个 token 的嵌入维度 (embedding dimension)
            num_heads: 注意力头的数量
            dropout: Dropout 概率，用于防止过拟合
        """
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # 定义 Q, K, V 的线性映射层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 注意力输出后的线性层
        self.fc = nn.Linear(embed_dim, embed_dim)

        # dropout 层（用于注意力权重和输出）
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, embed_dim]
        输出:
            输出张量形状相同: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.num_heads * self.head_dim, "输入维度与多头配置不匹配"

        # 1️⃣ 线性变换得到 Q, K, V
        Q = self.query(x)  # [batch, seq_len, embed_dim]
        K = self.key(x)
        V = self.value(x)

        # 2️⃣ 拆分成多个头并转置： [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3️⃣ 计算注意力得分：Q × K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # 4️⃣ 对得分进行 Softmax，得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 5️⃣ 对注意力权重进行 dropout（训练时随机屏蔽部分连接）
        attn_weights = self.attn_dropout(attn_weights)

        # 6️⃣ 计算加权和：Attention(Q, K, V) = softmax(QKᵀ/√d) V
        attended_values = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # 7️⃣ 将多头拼接回来: [batch, seq_len, embed_dim]
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 8️⃣ 输出线性层 + dropout + 残差连接
        out = self.fc(attended_values)
        out = self.output_dropout(out)
        out = out + x  # 残差连接（Residual connection）

        return out


# 示例验证
x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
attn = MultiHeadSelfAttention(embed_dim=16, num_heads=4)
out = attn(x)

print("输入形状:", x.shape)
print("输出形状:", out.shape)