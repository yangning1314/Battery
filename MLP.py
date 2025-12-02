import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# 0. 配置区域
# ==========================================
DATA_FOLDER = 'Batch-1'
SAVE_IMG_PATH = 'multi_battery_result.png'
RATED_CAPACITY = 2.0

# 训练参数 (为 PINN 优化)
EPOCHS = 1000
LR = 0.005
LAMBDA_PHY = 0.05  # 物理权重
LAMBDA_MONO = 0.1  # 单调性权重 (在新电池上这很重要)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 多文件数据加载模块
# ==========================================
def load_battery_data(folder, battery_ids):
    """
    读取指定 ID 列表的电池数据，并合并
    battery_ids: list, e.g., [1, 2, 3, 4, 5, 6]
    """
    all_X = []
    all_Y = []

    print(f"正在加载电池 ID: {battery_ids} ...")

    for bid in battery_ids:
        filename = f"2C_battery-{bid}.csv"
        path = os.path.join(folder, filename)

        try:
            # 兼容不同分隔符
            df = pd.read_csv(path)
            if df.shape[1] < 2: df = pd.read_csv(path, sep='\t')

            # 清洗
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # 提取 X (特征) 和 Y (SOH)
            X = df.iloc[:, :-1].values.astype(np.float32)
            Y = (df.iloc[:, -1].values / RATED_CAPACITY).astype(np.float32)

            all_X.append(X)
            all_Y.append(Y)

        except Exception as e:
            print(f"无法读取 {filename}: {e}")
            continue

    # 垂直合并所有数据
    if len(all_X) == 0: raise ValueError("没有加载到任何数据！")

    X_concat = np.concatenate(all_X, axis=0)
    Y_concat = np.concatenate(all_Y, axis=0)

    return X_concat, Y_concat


# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
class BackboneNet(nn.Module):
    def __init__(self, input_dim):
        super(BackboneNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # 增加神经元以处理更多数据
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x): return self.net(x)


class DynamicsNet(nn.Module):
    def __init__(self, input_dim):
        super(DynamicsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x): return self.net(x)


# ==========================================
# 3. 训练逻辑
# ==========================================
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).view(-1, 1)

    def __len__(self): return len(self.X) - 1

    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.X[idx + 1]


def train_mlp(X, y, input_dim):
    print(">>> [Baseline] Training MLP on Batteries 1-6...")
    model = BackboneNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    loader = DataLoader(BatteryDataset(X, y), batch_size=64, shuffle=True)  # Batch加大

    for epoch in range(EPOCHS):
        model.train()
        for x_k, y_k, _ in loader:
            x_k, y_k = x_k.to(DEVICE), y_k.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_k)
            loss = nn.MSELoss()(pred, y_k)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def train_pinn(X, y, input_dim):
    print(">>> [Proposed] Training PINN on Batteries 1-6...")
    net_f = BackboneNet(input_dim).to(DEVICE)
    net_g = DynamicsNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(list(net_f.parameters()) + list(net_g.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    loader = DataLoader(BatteryDataset(X, y), batch_size=64, shuffle=True)

    for epoch in range(EPOCHS):
        net_f.train();
        net_g.train()
        for x_k, y_k, x_k1 in loader:
            x_k, y_k, x_k1 = x_k.to(DEVICE), y_k.to(DEVICE), x_k1.to(DEVICE)
            optimizer.zero_grad()

            # Loss 1: Data
            u_k = net_f(x_k)
            loss_data = nn.MSELoss()(u_k, y_k)

            # Loss 2: Physics
            u_k1_f = net_f(x_k1)
            inp_g = torch.cat([u_k.detach(), x_k], dim=1)
            delta = net_g(inp_g)
            u_k1_phy = u_k + delta
            loss_phy = nn.MSELoss()(u_k1_f, u_k1_phy)

            # Loss 3: Mono
            loss_mono = torch.mean(torch.relu(delta))

            loss = loss_data + LAMBDA_PHY * loss_phy + LAMBDA_MONO * loss_mono
            loss.backward()
            optimizer.step()
        scheduler.step()
    return net_f


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 数据划分策略
    # 训练集: 电池 1, 2, 3, 4, 5, 6
    # 测试集: 电池 7, 8 (完全没见过的数据，考验泛化能力)
    train_ids = [1, 2, 3, 4, 5, 6]
    test_ids = [7, 8]

    # 2. 加载合并数据
    X_train_raw, y_train = load_battery_data(DATA_FOLDER, train_ids)
    X_test_raw, y_test = load_battery_data(DATA_FOLDER, test_ids)

    # 3. 归一化 (注意：必须只在训练集上fit，应用到测试集)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)  # transform ONLY

    input_dim = X_train.shape[1]
    print(f"特征维度: {input_dim}")
    print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")

    # 4. 训练
    model_mlp = train_mlp(X_train, y_train, input_dim)
    model_pinn = train_pinn(X_train, y_train, input_dim)

    # 5. 评估
    model_mlp.eval();
    model_pinn.eval()
    with torch.no_grad():
        inputs_test = torch.tensor(X_test).to(DEVICE)
        pred_mlp = model_mlp(inputs_test).cpu().numpy()
        pred_pinn = model_pinn(inputs_test).cpu().numpy()

    rmse_mlp = np.sqrt(mean_squared_error(y_test, pred_mlp))
    rmse_pinn = np.sqrt(mean_squared_error(y_test, pred_pinn))
    imp = ((rmse_mlp - rmse_pinn) / rmse_mlp) * 100

    print("\n" + "=" * 40)
    print(f"测试集结果 (Batteries {test_ids})")
    print("=" * 40)
    print(f"MLP RMSE : {rmse_mlp:.5f}")
    print(f"PINN RMSE: {rmse_pinn:.5f}")
    print(f"提升比例 : {imp:.2f}%")
    print("=" * 40)

    # 6. 绘图
    # 为了图表好看，我们只画出 测试集中 某一个电池 的连续曲线
    # 比如只画测试数据的后半部分（对应电池 8）
    # 因为直接画所有测试集，两个电池的数据拼接处会断崖，不好看

    plt.figure(figsize=(10, 6))

    # 简单的切分一下用于展示 (假设数据量大致均分，取后半段展示电池8)
    display_start = int(len(y_test) * 0.5)

    cycle_idx = np.arange(len(y_test[display_start:]))

    plt.plot(cycle_idx, y_test[display_start:], 'k-', label='True SOH (Battery 8)', linewidth=2, alpha=0.6)
    plt.plot(cycle_idx, pred_mlp[display_start:], 'b:', label=f'MLP (RMSE={rmse_mlp:.4f})', linewidth=1.5)
    plt.plot(cycle_idx, pred_pinn[display_start:], 'r--', label=f'PINN (RMSE={rmse_pinn:.4f})', linewidth=2)

    plt.title(f'Generalization Test on Unseen Battery (No.{test_ids[-1]})')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(SAVE_IMG_PATH, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {SAVE_IMG_PATH}")
    plt.show()