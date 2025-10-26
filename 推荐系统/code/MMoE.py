import torch
import torch.nn as nn
import numpy as np


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: 输入特征维度
        output_dim: expert 输出的 embedding 维度（expert_dim）
        """
        super(Expert, self).__init__()

        p = 0.0
        expert_hidden_layers = [64, 32]

        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(expert_hidden_layers[0], expert_hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(expert_hidden_layers[1], output_dim),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.expert_layer(x)


class Expert_Gate(nn.Module):
    def __init__(self, feature_dim, expert_dim, n_expert, n_task, use_gate=True):
        """
        feature_dim: 输入特征维度
        expert_dim: 每个 expert 输出维度
        n_expert: 专家数量
        n_task: 任务数 (也就是要输出多少个不同的门控组合)
        use_gate: 是否使用门控（如果 False，就简单平均所有 expert）
        """
        super(Expert_Gate, self).__init__()

        self.n_task = n_task
        self.use_gate = use_gate
        self.n_expert = n_expert
        self.expert_dim = expert_dim

        # 专家网络们: n_expert 个 Expert
        self.expert_layers = nn.ModuleList([
            Expert(feature_dim, expert_dim) for _ in range(n_expert)
        ])

        # 门控网络们: n_task 个 gate，每个 gate 给一组专家分配权重
        if use_gate:
            self.gate_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, n_expert),
                    nn.Softmax(dim=1)
                ) for _ in range(n_task)
            ])
        else:
            self.gate_layers = None  # 不用 gate

    def forward(self, x):
        """
        x: (bs, feature_dim)
        return:
            if use_gate == True:
                list of length n_task, each item shape (bs, expert_dim)
            else:
                tensor shape (bs, expert_dim)  # (简单平均)
        """

        # 1. 计算所有专家输出
        # E_net_list: n_expert 个 [bs, expert_dim]
        E_net_list = [expert(x) for expert in self.expert_layers]

        if self.use_gate:
            # 堆成一个 tensor: (bs, n_expert, expert_dim)
            E_net = torch.stack(E_net_list, dim=1)

            # 对每个任务分别做门控加权
            towers = []
            for gate in self.gate_layers:
                gate_weight = gate(x)                      # (bs, n_expert)
                gate_weight = gate_weight.unsqueeze(2)     # (bs, n_expert, 1)

                # batch 矩阵乘: (bs, expert_dim, n_expert) x (bs, n_expert, 1)
                mixed = torch.bmm(
                    E_net.transpose(1, 2),                 # (bs, expert_dim, n_expert)
                    gate_weight                            # (bs, n_expert, 1)
                )                                          # -> (bs, expert_dim, 1)

                mixed = mixed.squeeze(2)                   # (bs, expert_dim)
                towers.append(mixed)

            return towers  # list 长度 = n_task

        else:
            # 不使用 gate，直接对专家平均
            avg = sum(E_net_list) / len(E_net_list)        # (bs, expert_dim)
            return avg


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) for two tasks
    feature_dim: 输入特征维度
    expert_dim: expert 输出维度（传给 task tower 的特征维度）
    n_expert: 专家数量
    n_task: 任务数
    use_gate: 是否使用 gate
    """
    def __init__(self, feature_dim, expert_dim, n_expert, n_task, use_gate=True):
        super(MMoE, self).__init__()

        self.use_gate = use_gate
        self.n_task = n_task

        self.expert_gate = Expert_Gate(
            feature_dim=feature_dim,
            expert_dim=expert_dim,
            n_expert=n_expert,
            n_task=n_task,
            use_gate=use_gate
        )

        # 下面我们假定有两个任务 (task1 / task2)
        # 如果以后你要扩展到 n_task>2，只需要把 tower 改成 ModuleList 动态生成
        hidden_layer1 = [64, 32]
        p1 = 0.0
        self.tower1 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Dropout(p1),

            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Dropout(p1),

            nn.Linear(hidden_layer1[1], 1)
        )

        hidden_layer2 = [64, 32]
        p2 = 0.0
        self.tower2 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Dropout(p2),

            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Dropout(p2),

            nn.Linear(hidden_layer2[1], 1)
        )

    def forward(self, x):
        towers = self.expert_gate(x)

        if self.use_gate:
            # towers 是 list: [task1_embed, task2_embed]
            task1_feat = towers[0]
            task2_feat = towers[1]
        else:
            # towers 是 tensor: 共享一个平均的特征
            task1_feat = towers
            task2_feat = towers

        out1 = self.tower1(task1_feat)   # (bs, 1)
        out2 = self.tower2(task2_feat)   # (bs, 1)

        return out1, out2


if __name__ == "__main__":
    # 假设我们有 112 维输入特征
    feature_dim = 112
    expert_dim = 32
    n_expert = 4
    n_task = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MMoE(
        feature_dim=feature_dim,
        expert_dim=expert_dim,
        n_expert=n_expert,
        n_task=n_task,
        use_gate=True
    ).to(device)

    # 随机一批样本(batch_size=16)
    x_dummy = torch.randn(16, feature_dim).to(device)

    # 前向
    y1_pred, y2_pred = model(x_dummy)

    print("* y1_pred shape:", y1_pred.shape)  # 期望: [16, 1]
    print("* y2_pred shape:", y2_pred.shape)  # 期望: [16, 1]

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print("* number of parameters:", n_params)
