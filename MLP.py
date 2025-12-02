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
# 0. 配置
# ==========================================
CSV_PATH = 'battery_data.csv'
SAVE_IMG_PATH = 'comparison_result.png'
RATED_CAPACITY = 2.0
EPOCHS = 500
LR = 0.001


# ==========================================
# 1. 数据加载 (同之前，含清洗)
# ==========================================
def load_data(csv_path):
    print(f"读取数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2: df = pd.read_csv(csv_path, sep='\t')
    except:
        raise ValueError("读取失败")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values / RATED_CAPACITY).astype(np.float32)
    return X, y


# ==========================================
# 2. 定义模型 (MLP 和 PINN 共享相同的骨干)
# ==========================================
class BackboneNet(nn.Module):
    """ 两个模型共用的 SOH 估计网络 """

    def __init__(self, input_dim):
        super(BackboneNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x): return self.net(x)


class DynamicsNet(nn.Module):
    """ 仅 PINN 使用的物理动力学网络 """

    def __init__(self, input_dim):
        super(DynamicsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x): return self.net(x)


# ==========================================
# 3. 训练辅助函数
# ==========================================
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).view(-1, 1)

    def __len__(self): return len(self.X) - 1

    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.X[idx + 1]


def train_mlp(X_train, y_train, input_dim, device):
    print(">>> 正在训练 Baseline (Pure MLP)...")
    model = BackboneNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    dataset = BatteryDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for x_k, y_k, _ in loader:  # MLP 不需要 x_k+1
            x_k, y_k = x_k.to(device), y_k.to(device)
            optimizer.zero_grad()
            pred = model(x_k)
            loss = nn.MSELoss()(pred, y_k)  # 只有 Data Loss
            loss.backward()
            optimizer.step()
    return model


def train_pinn(X_train, y_train, input_dim, device):
    print(">>> 正在训练 Proposed Method (PINN)...")
    net_f = BackboneNet(input_dim).to(device)
    net_g = DynamicsNet(input_dim).to(device)
    optimizer = optim.Adam(list(net_f.parameters()) + list(net_g.parameters()), lr=LR)

    dataset = BatteryDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(EPOCHS):
        net_f.train();
        net_g.train()
        for x_k, y_k, x_k1 in loader:
            x_k, y_k, x_k1 = x_k.to(device), y_k.to(device), x_k1.to(device)
            optimizer.zero_grad()

            # 1. Data Loss
            u_k = net_f(x_k)
            loss_data = nn.MSELoss()(u_k, y_k)

            # 2. Physics Loss
            u_k1_f = net_f(x_k1)
            inp_g = torch.cat([u_k.detach(), x_k], dim=1)  # 物理输入
            delta = net_g(inp_g)
            u_k1_phy = u_k + delta
            loss_phy = nn.MSELoss()(u_k1_f, u_k1_phy)

            # 3. Mono Loss
            loss_mono = torch.mean(torch.relu(delta))

            loss = loss_data + 0.2 * loss_phy + 0.1 * loss_mono
            loss.backward()
            optimizer.step()
    return net_f


# ==========================================
# 4. 主流程
# ==========================================
if __name__ == "__main__":
    # A. 数据准备
    X_raw, Y_soh = load_data(CSV_PATH)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    split_idx = int(len(X_scaled) * 0.8)
    X_train, y_train = X_scaled[:split_idx], Y_soh[:split_idx]
    X_test, y_test = X_scaled[split_idx:], Y_soh[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    # B. 分别训练两个模型
    model_mlp = train_mlp(X_train, y_train, input_dim, device)
    model_pinn = train_pinn(X_train, y_train, input_dim, device)

    # C. 预测与评估
    model_mlp.eval();
    model_pinn.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(X_scaled).to(device)
        pred_mlp = model_mlp(input_tensor).cpu().numpy()
        pred_pinn = model_pinn(input_tensor).cpu().numpy()

    # 计算 RMSE (仅在测试集)
    rmse_mlp = np.sqrt(mean_squared_error(Y_soh[split_idx:], pred_mlp[split_idx:]))
    rmse_pinn = np.sqrt(mean_squared_error(Y_soh[split_idx:], pred_pinn[split_idx:]))

    print(f"\n=== 最终对比结果 (Test Set) ===")
    print(f"Pure MLP RMSE: {rmse_mlp:.5f}")
    print(f"PINN     RMSE: {rmse_pinn:.5f}")
    print(f"PINN 提升比例: {((rmse_mlp - rmse_pinn) / rmse_mlp) * 100:.2f}%")

    # D. 绘图保存
    plt.figure(figsize=(10, 6))
    plt.plot(Y_soh, 'k-', label='True SOH', linewidth=1.5, alpha=0.6)

    # 绘制 MLP 结果 (蓝色虚线)
    plt.plot(pred_mlp, 'b:', label=f'Pure MLP (RMSE={rmse_mlp:.4f})', linewidth=1.5)

    # 绘制 PINN 结果 (红色实线)
    plt.plot(pred_pinn, 'r--', label=f'PINN (RMSE={rmse_pinn:.4f})', linewidth=2)

    plt.axvline(x=split_idx, color='g', linestyle='-.', label='Train/Test Split')

    plt.title('Comparison: Pure MLP vs. PINN (Physics-Informed)')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)

    plt.savefig(SAVE_IMG_PATH, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {SAVE_IMG_PATH}")
    plt.show()