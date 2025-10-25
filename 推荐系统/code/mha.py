import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        """
        多头注意力机制（Multi-Head Attention）
        :param num_heads: 注意力头的数量
        :param d_model: 输入与输出的特征维度
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义线性层，生成 Q、K、V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出线性层
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: [batch_size, seq_len, d_model]
        :param mask: 注意力mask，可选
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        # 1️⃣ 计算 Q、K、V 并拆分为多头
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2️⃣ 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        # 3️⃣ 应用 mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4️⃣ softmax 归一化得到注意力权重
        attn_scores = F.softmax(scores, dim=-1)

        # 5️⃣ 加权求和得到输出
        output = torch.matmul(attn_scores, V)

        # 6️⃣ 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 7️⃣ 线性映射回原维度
        output = self.W_O(output)
        return output


# ========================================
# ✅ 测试用例
# ========================================
if __name__ == "__main__":
    torch.manual_seed(42)  # 固定随机种子方便复现

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    # 构造随机输入 [batch_size, seq_len, d_model]
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建多头注意力层
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)

    # 前向传播
    output = mha(x)

    print("✅ 输入形状:", x.shape)
    print("✅ 输出形状:", output.shape)
    print("✅ 输出示例:\n", output[0])