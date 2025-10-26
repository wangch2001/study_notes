import torch
import torch.nn as nn
import numpy as np

'''---------------------------------
    Expert 子网络
---------------------------------'''
class Expert_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert_net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


'''---------------------------------
    特征提取层 Extraction_Network
---------------------------------'''
class Extraction_Network(nn.Module):
    """
    FeatureDim: 输入特征维度
    ExpertOutDim: 每个Expert输出维度
    TaskExpertNum: 每个任务的专属Expert数
    CommonExpertNum: 共享Expert数
    GateNum: gate数 (2表示最后一层, 3表示中间层)
    """

    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum):
        super(Extraction_Network, self).__init__()

        self.GateNum = GateNum
        self.TaskExpertNum = TaskExpertNum
        self.CommonExpertNum = CommonExpertNum

        '''两个任务模块，一个共享模块'''
        self.n_task = 2
        self.n_share = 1

        # --- TaskA Experts ---
        self.Experts_A = nn.ModuleList(
            [Expert_net(FeatureDim, ExpertOutDim) for _ in range(TaskExpertNum)]
        )

        # --- Shared Experts ---
        self.Experts_Shared = nn.ModuleList(
            [Expert_net(FeatureDim, ExpertOutDim) for _ in range(CommonExpertNum)]
        )

        # --- TaskB Experts ---
        self.Experts_B = nn.ModuleList(
            [Expert_net(FeatureDim, ExpertOutDim) for _ in range(TaskExpertNum)]
        )

        # --- Task Gates ---
        self.Task_Gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FeatureDim, TaskExpertNum + CommonExpertNum),
                nn.Softmax(dim=1)
            ) for _ in range(self.n_task)
        ])

        # --- Shared Gate ---
        if GateNum == 3:
            self.Shared_Gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(FeatureDim, 2 * TaskExpertNum + CommonExpertNum),
                    nn.Softmax(dim=1)
                ) for _ in range(self.n_share)
            ])
        else:
            self.Shared_Gates = None

    def forward(self, x_A, x_S, x_B):

        # --- Experts_A 输出 ---
        Experts_A_Out = torch.stack([expert(x_A) for expert in self.Experts_A], dim=1)  # (bs, TaskExpertNum, ExpertOutDim)

        # --- Shared Experts 输出 ---
        Experts_Shared_Out = torch.stack([expert(x_S) for expert in self.Experts_Shared], dim=1)  # (bs, CommonExpertNum, ExpertOutDim)

        # --- Experts_B 输出 ---
        Experts_B_Out = torch.stack([expert(x_B) for expert in self.Experts_B], dim=1)  # (bs, TaskExpertNum, ExpertOutDim)

        # --- Gate A 权重 ---
        Gate_A = self.Task_Gates[0](x_A)  # (bs, TaskExpertNum + CommonExpertNum)

        # --- Gate Shared 权重 ---
        if self.GateNum == 3:
            Gate_Shared = self.Shared_Gates[0](x_S)  # (bs, 2*TaskExpertNum + CommonExpertNum)

        # --- Gate B 权重 ---
        Gate_B = self.Task_Gates[1](x_B)  # (bs, TaskExpertNum + CommonExpertNum)

        # --- GateA 输出 ---
        g = Gate_A.unsqueeze(2)  # (bs, TaskExpertNum+CommonExpertNum, 1)
        experts = torch.cat([Experts_A_Out, Experts_Shared_Out], dim=1)
        Gate_A_Out = torch.bmm(experts.transpose(1, 2), g).squeeze(2)  # (bs, ExpertOutDim)

        # --- GateShared 输出 ---
        if self.GateNum == 3:
            g = Gate_Shared.unsqueeze(2)
            experts = torch.cat([Experts_A_Out, Experts_Shared_Out, Experts_B_Out], dim=1)
            Gate_Shared_Out = torch.bmm(experts.transpose(1, 2), g).squeeze(2)
        else:
            Gate_Shared_Out = None

        # --- GateB 输出 ---
        g = Gate_B.unsqueeze(2)
        experts = torch.cat([Experts_B_Out, Experts_Shared_Out], dim=1)
        Gate_B_Out = torch.bmm(experts.transpose(1, 2), g).squeeze(2)

        if self.GateNum == 3:
            return Gate_A_Out, Gate_Shared_Out, Gate_B_Out
        else:
            return Gate_A_Out, Gate_B_Out


'''---------------------------------
    主体 PLE 网络
---------------------------------'''
class PLE(nn.Module):
    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, n_task=2):
        super(PLE, self).__init__()

        # 第一层 (底层) Extraction + 共享层
        self.Extraction_layer1 = Extraction_Network(
            FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=3
        )

        # 第二层 CGC 模块
        self.CGC = Extraction_Network(
            ExpertOutDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=2
        )

        # --- Task A Tower ---
        hidden_layer1 = [64, 32]
        self.tower1 = nn.Sequential(
            nn.Linear(ExpertOutDim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer1[1], 1)
        )

        # --- Task B Tower ---
        hidden_layer2 = [64, 32]
        self.tower2 = nn.Sequential(
            nn.Linear(ExpertOutDim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer2[1], 1)
        )

    def forward(self, x):
        Output_A, Output_Shared, Output_B = self.Extraction_layer1(x, x, x)
        Gate_A_Out, Gate_B_Out = self.CGC(Output_A, Output_Shared, Output_B)

        out1 = self.tower1(Gate_A_Out)
        out2 = self.tower2(Gate_B_Out)

        return out1, out2


# ----------------------------
# ✅ 测试代码
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.randn(128, 20).to(device)  # 模拟输入 (batch=128, 特征=20)
    Model = PLE(FeatureDim=20, ExpertOutDim=64, TaskExpertNum=2, CommonExpertNum=1).to(device)

    optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
    loss_func = nn.MSELoss().to(device)

    nParams = sum(p.numel() for p in Model.parameters())
    print('* number of parameters: %d' % nParams)

    y1, y2 = Model(X_train)
    print("Output shapes:", y1.shape, y2.shape)
