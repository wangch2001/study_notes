import torch
import torch.nn as nn
import numpy as np

'''---------------------------------
    Expert 子网络
    这是构成 PLE 结构的基本单元。
    每个 Expert 都是一个简单的多层感知机（MLP）。
---------------------------------'''


class Expert_net(nn.Module):
    """
    单个 Expert 网络的定义。
    它由一个线性层和一个ReLU激活函数组成。
    """

    def __init__(self, input_dim, output_dim):
        """
        初始化 Expert 网络。
        :param input_dim: 输入特征维度
        :param output_dim: Expert 的输出维度
        """
        super(Expert_net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # 线性变换
            nn.ReLU()  # ReLU 激活
        )

    def forward(self, x):
        """
        Expert 的前向传播。
        :param x: 输入张量，形状 (batch_size, input_dim)
        :return: 输出张量，形状 (batch_size, output_dim)
        """
        return self.layer(x)


'''---------------------------------
    特征提取层 Extraction_Network
    这是 PLE 模型中的一个“层”（Level）。
    它包含了任务A的专家、任务B的专家、共享专家，以及对应的门控网络（Gating Network）。
---------------------------------'''


class Extraction_Network(nn.Module):
    """
    定义 PLE 中的一个提取层（CGC - Customized Gate Control）。

    参数:
    FeatureDim:      输入特征维度
    ExpertOutDim:    每个Expert的输出维度
    TaskExpertNum:   每个任务的专属Expert数量
    CommonExpertNum: 共享Expert的数量
    GateNum:         门控网络数量 (2 表示这是最后一层提取层，只输出给Tower；
                                 3 表示这是中间层，会输出给下一层的 任务A、共享、任务B)
    """

    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum):
        super(Extraction_Network, self).__init__()

        self.GateNum = GateNum  # 门控数量
        self.TaskExpertNum = TaskExpertNum  # 任务特定专家数
        self.CommonExpertNum = CommonExpertNum  # 共享专家数

        '''两个任务模块，一个共享模块'''
        # 注意：这里硬编码为2个任务 (A 和 B) 和 1个共享组件
        self.n_task = 2  # 任务数量
        self.n_share = 1  # 共享组件数量

        # --- TaskA Experts (任务A的专属专家) ---
        self.Experts_A = nn.ModuleList(
            [Expert_net(FeatureDim, ExpertOutDim) for _ in range(TaskExpertNum)]
        )

        # --- Shared Experts (共享专家) ---
        self.Experts_Shared = nn.ModuleList(
            [Expert_net(FeatureDim, ExpertOutDim) for _ in range(CommonExpertNum)]
        )

        # --- TaskB Experts (任务B的专属专家) ---
        self.Experts_B = nn.ModuleList(
            [Expert_net(FeatureDim, ExpertOutDim) for _ in range(TaskExpertNum)]
        )

        # --- Task Gates (任务A 和 任务B 的门控网络) ---
        # 每个任务门控网络负责对其 "专属专家" 和 "共享专家" 的输出进行加权求和
        self.Task_Gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FeatureDim, TaskExpertNum + CommonExpertNum),  # 门控的输出维度 = 专属专家数 + 共享专家数
                nn.Softmax(dim=1)  # Softmax 按行（样本）归一化，得到权重
            ) for _ in range(self.n_task)  # 创建 n_task 个 (这里是2个) 任务门控
        ])

        # --- Shared Gate (共享门控网络) ---
        # 仅当 GateNum=3 (即中间层) 时才需要共享门控
        if GateNum == 3:
            # 共享门控网络负责对 "所有专家" (A, B, 共享) 的输出进行加权求和
            self.Shared_Gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(FeatureDim, 2 * TaskExpertNum + CommonExpertNum),  # 门控输出维度 = A专家数 + B专家数 + 共享专家数
                    nn.Softmax(dim=1)
                ) for _ in range(self.n_share)  # 创建 n_share 个 (这里是1个) 共享门控
            ])
        else:
            self.Shared_Gates = None  # 顶层不需要共享门控

    def forward(self, x_A, x_S, x_B):
        """
        Extraction_Network 的前向传播。
        :param x_A: 供给任务A专家和门控的输入，形状 (bs, FeatureDim)
        :param x_S: 供给共享专家和门控的输入，形状 (bs, FeatureDim)
        :param x_B: 供给任务B专家和门控的输入，形状 (bs, FeatureDim)
        :return: 门控网络的加权输出
        """

        # --- 1. 计算所有专家的输出 ---

        # Experts_A 输出
        # 将 x_A 分别送入 TaskExpertNum 个 专家A
        # 结果堆叠，形状: (bs, TaskExpertNum, ExpertOutDim)
        Experts_A_Out = torch.stack([expert(x_A) for expert in self.Experts_A], dim=1)

        # Shared Experts 输出
        # 将 x_S 分别送入 CommonExpertNum 个 共享专家
        # 结果堆叠，形状: (bs, CommonExpertNum, ExpertOutDim)
        Experts_Shared_Out = torch.stack([expert(x_S) for expert in self.Experts_Shared], dim=1)

        # Experts_B 输出
        # 将 x_B 分别送入 TaskExpertNum 个 专家B
        # 结果堆叠，形状: (bs, TaskExpertNum, ExpertOutDim)
        Experts_B_Out = torch.stack([expert(x_B) for expert in self.Experts_B], dim=1)

        # --- 2. 计算所有门控的权重 ---

        # Gate A 权重
        # 形状: (bs, TaskExpertNum + CommonExpertNum)
        Gate_A = self.Task_Gates[0](x_A)

        # Gate Shared 权重 (如果是中间层)
        if self.GateNum == 3:
            # 形状: (bs, 2*TaskExpertNum + CommonExpertNum)
            Gate_Shared = self.Shared_Gates[0](x_S)

        # Gate B 权重
        # 形状: (bs, TaskExpertNum + CommonExpertNum)
        Gate_B = self.Task_Gates[1](x_B)

        # --- 3. 计算门控加权输出 ---

        # --- GateA 输出 (任务A的加权输出) ---
        g = Gate_A.unsqueeze(2)  # 形状: (bs, TaskExpertNum+CommonExpertNum, 1)
        # 门控A 只选择 专家A 和 共享专家
        experts = torch.cat([Experts_A_Out, Experts_Shared_Out],
                            dim=1)  # 形状: (bs, TaskExpertNum+CommonExpertNum, ExpertOutDim)
        # bmm: 批量矩阵乘法，实现加权求和
        # (bs, ExpertOutDim, TaskExpertNum+CommonExpertNum) bmm (bs, TaskExpertNum+CommonExpertNum, 1) -> (bs, ExpertOutDim, 1)
        Gate_A_Out = torch.bmm(experts.transpose(1, 2), g).squeeze(2)  # 最终形状: (bs, ExpertOutDim)

        # --- GateShared 输出 (共享加权输出) ---
        if self.GateNum == 3:
            g = Gate_Shared.unsqueeze(2)  # 形状: (bs, 2*TaskExpertNum+CommonExpertNum, 1)
            # 共享门控 选择 所有专家
            experts = torch.cat([Experts_A_Out, Experts_Shared_Out, Experts_B_Out],
                                dim=1)  # 形状: (bs, 2*TaskExpertNum+CommonExpertNum, ExpertOutDim)
            # (bs, ExpertOutDim, 2*T+C) bmm (bs, 2*T+C, 1) -> (bs, ExpertOutDim, 1)
            Gate_Shared_Out = torch.bmm(experts.transpose(1, 2), g).squeeze(2)  # 最终形状: (bs, ExpertOutDim)
        else:
            Gate_Shared_Out = None  # 顶层无共享输出

        # --- GateB 输出 (任务B的加权输出) ---
        g = Gate_B.unsqueeze(2)  # 形状: (bs, TaskExpertNum+CommonExpertNum, 1)
        # 门控B 只选择 专家B 和 共享专家
        experts = torch.cat([Experts_B_Out, Experts_Shared_Out],
                            dim=1)  # 形状: (bs, TaskExpertNum+CommonExpertNum, ExpertOutDim)
        # (bs, ExpertOutDim, TaskExpertNum+CommonExpertNum) bmm (bs, TaskExpertNum+CommonExpertNum, 1) -> (bs, ExpertOutDim, 1)
        Gate_B_Out = torch.bmm(experts.transpose(1, 2), g).squeeze(2)  # 最终形状: (bs, ExpertOutDim)

        # 根据层类型返回
        if self.GateNum == 3:
            return Gate_A_Out, Gate_Shared_Out, Gate_B_Out
        else:
            return Gate_A_Out, Gate_B_Out


'''---------------------------------
    主体 PLE 网络
    (Progressive Layered Extraction)
    将多个 Extraction_Network 堆叠起来，并连接上层的 Tower。
---------------------------------'''


class PLE(nn.Module):
    """
    完整的 PLE 模型。

    在这个实现中，它堆叠了两个 Extraction_Network 层：
    1. 底层 (Extraction_layer1): GateNum=3，有 任务A、任务B、共享 三个输出。
    2. 顶层 (CGC): GateNum=2，只有 任务A、任务B 两个输出，分别送入各自的 Tower。

    参数: (同 Extraction_Network)
    FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum
    n_task: 任务数量 (此处未使用，因为 Extraction_Network 中硬编码为2)
    """

    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, n_task=2):
        super(PLE, self).__init__()

        # 第一层 (底层) Extraction + 共享层
        # GateNum=3 表示这是一个中间层，它会为 任务A、共享、任务B 分别产生一个输出
        self.Extraction_layer1 = Extraction_Network(
            FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=3
        )

        # 第二层 (顶层) CGC 模块
        # GateNum=2 表示这是一个顶层，它只为 任务A、任务B 产生输出，准备送入Tower
        # 它的输入维度 (ExpertOutDim) 必须等于上一层 (Extraction_layer1) 的输出维度 (ExpertOutDim)
        self.CGC = Extraction_Network(
            ExpertOutDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=2
        )

        # --- Task A Tower (任务A的塔) ---
        # 接收来自顶层 CGC 的 任务A 输出，并进行最终预测
        hidden_layer1 = [64, 32]  # 塔的隐藏层维度
        self.tower1 = nn.Sequential(
            nn.Linear(ExpertOutDim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer1[1], 1)  # 最终输出1维（例如回归任务）
        )

        # --- Task B Tower (任务B的塔) ---
        # 接收来自顶层 CGC 的 任务B 输出，并进行最终预测
        hidden_layer2 = [64, 32]  # 塔的隐藏层维度
        self.tower2 = nn.Sequential(
            nn.Linear(ExpertOutDim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer2[1], 1)  # 最终输出1维
        )

    def forward(self, x):
        """
        PLE 模型的前向传播。
        :param x: 原始输入特征，形状 (bs, FeatureDim)
        :return: 两个任务的输出 (out1, out2)
        """

        # --- 第一层 ---
        # 原始输入 x 同时作为 任务A、共享、任务B 的输入
        Output_A, Output_Shared, Output_B = self.Extraction_layer1(x, x, x)

        # --- 第二层 ---
        # 将第一层的 A、Shared、B 输出，分别作为第二层 A、Shared、B 的输入
        # 注意：CGC (GateNum=2) 只返回两个输出
        Gate_A_Out, Gate_B_Out = self.CGC(Output_A, Output_Shared, Output_B)

        # --- Towers ---
        # 将顶层提取的特征送入各自的塔
        out1 = self.tower1(Gate_A_Out)
        out2 = self.tower2(Gate_B_Out)

        # 返回两个任务的最终预测值
        return out1, out2


# ----------------------------
# ✅ 测试代码
# ----------------------------
if __name__ == "__main__":
    # 1. 设置设备 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 模拟输入数据
    # batch_size=128, 特征维度=20
    X_train = torch.randn(128, 20).to(device)

    # 3. 实例化 PLE 模型
    # FeatureDim=20 (必须与输入匹配)
    # ExpertOutDim=64 (每个Expert的输出维度)
    # TaskExpertNum=2 (每个任务有2个专属Expert)
    # CommonExpertNum=1 (有1个共享Expert)
    Model = PLE(FeatureDim=20, ExpertOutDim=64, TaskExpertNum=2, CommonExpertNum=1).to(device)

    # 4. (可选) 定义优化器和损失函数
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
    loss_func = nn.MSELoss().to(device)  # 假设是回归任务

    # 5. (可选) 计算模型参数量
    nParams = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print('* 模型参数量 (Number of parameters): %d' % nParams)

    # 6. 执行一次前向传播
    y1, y2 = Model(X_train)

    # 7. 打印输出形状进行验证
    # 应该输出 (128, 1) 和 (128, 1)
    print("Output shapes:", y1.shape, y2.shape)