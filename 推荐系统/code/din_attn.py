import torch
import torch.nn as nn
import torch.nn.functional as F


class DINAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # MLP 用于计算注意力分数: 输入是 [e_i, a, e_i * a] -> 3*embed_dim
        self.fc = nn.Sequential(
            nn.Linear(3 * embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出一个标量分数
        )

    def forward(self, user_hist, target_item):
        """
        Args:
            user_hist: (B, T, D) 用户历史行为序列，B=batch, T=序列长度, D=embedding大小
            target_item: (B, D) 当前候选商品

        Returns:
            weighted_hist: (B, D) 加权后的用户兴趣向量
        """
        B, T, D = user_hist.shape

        # 扩展 target_item 到 (B, T, D)
        target_item_expanded = target_item.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)

        # 计算逐元素积: e_i * a
        element_wise_prod = user_hist * target_item_expanded  # (B, T, D)

        # 拼接: [e_i, a, e_i * a] -> (B, T, 3*D)
        attention_input = torch.cat([user_hist, target_item_expanded, element_wise_prod], dim=-1)

        # 通过 MLP 得到原始分数 -> (B, T, 1)
        scores = self.fc(attention_input)  # (B, T, 1)

        # Softmax 归一化 -> (B, T, 1)
        attn_weights = F.softmax(scores, dim=1)

        # 加权求和: (B, T, 1) * (B, T, D) -> (B, T, D) -> (B, D)
        weighted_hist = torch.sum(attn_weights * user_hist, dim=1)

        return weighted_hist


# ================== 测试 ==================
if __name__ == "__main__":
    embed_dim = 8
    model = DINAttention(embed_dim)

    # 模拟输入
    user_hist = torch.randn(2, 5, embed_dim)  # 2个用户，历史长度5
    target_item = torch.randn(2, embed_dim)  # 2个候选商品

    output = model(user_hist, target_item)

    print("输出形状:", output.shape)  # 应为 (2, 8)
    # 输出: torch.Size([2, 8])