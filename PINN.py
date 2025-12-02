import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ==========================================
# 0. 配置区域
# ==========================================
CSV_PATH = 'battery_data.csv'  # 你的数据文件名
IMAGE_SAVE_PATH = 'pinn_result.png'  # 图片保存的文件名
RATED_CAPACITY = 2.0


# ==========================================
# 1. 数据加载与清洗模块
# ==========================================

def load_and_process_data(csv_path):
    print(f"正在读取数据: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            df = pd.read_csv(csv_path, sep='\t')
    except Exception as e:
        raise ValueError(f"读取CSV失败: {e}")

    # --- [数据清洗] ---
    # 1. 替换 Inf 为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. 删除包含 NaN 的行
    if df.isnull().values.any():
        print("检测到坏点 (Inf/NaN)，正在剔除...")
        df.dropna(inplace=True)

    # --- 特征分离 ---
    X_raw = df.iloc[:, :-1].values
    capacity_raw = df.iloc[:, -1].values

    # 计算 SOH
    y_soh = capacity_raw / RATED_CAPACITY

    # 类型转换
    X_raw = X_raw.astype(np.float32)
    y_soh = y_soh.astype(np.float32)

    return X_raw, y_soh


# ==========================================
# 2. PINN 模型架构
# ==========================================

class PINN(nn.Module):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        self.net_f = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.net_g = nn.Sequential(
            nn.Linear(input_dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x): return self.net_f(x)

    def physics_forward(self, soh, x):
        return self.net_g(torch.cat([soh, x], dim=1))


# ==========================================
# 3. Dataset
# ==========================================

class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X) - 1

    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.X[idx + 1]


# ==========================================
# 4. 主程序
# ==========================================

if __name__ == "__main__":
    # 1. 加载并清洗
    X_raw, Y_soh = load_and_process_data(CSV_PATH)

    # 2. 归一化
    scaler = MinMaxScaler()
    try:
        X_scaled = scaler.fit_transform(X_raw)
    except ValueError as e:
        print("归一化失败，可能数据仍有异常:", e)
        exit()

    # 3. 划分训练/测试集
    split_idx = int(len(X_scaled) * 0.8)
    X_train, y_train = X_scaled[:split_idx], Y_soh[:split_idx]
    X_test, y_test = X_scaled[split_idx:], Y_soh[split_idx:]

    # 4. 准备 DataLoader
    dataset = BatteryDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 500
    lambda_phy = 0.2
    lambda_mono = 0.1

    # 5. 训练
    loss_history = []
    print("开始训练...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_k, y_k, x_k1 in train_loader:
            x_k, y_k, x_k1 = x_k.to(device), y_k.to(device), x_k1.to(device)
            optimizer.zero_grad()

            # Data Loss
            u_k = model(x_k)
            loss_data = nn.MSELoss()(u_k, y_k)

            # Physics Loss
            u_k1_f = model(x_k1)
            delta = model.physics_forward(u_k.detach(), x_k)
            u_k1_phy = u_k + delta
            loss_phy = nn.MSELoss()(u_k1_f, u_k1_phy)

            # Mono Loss
            loss_mono = torch.mean(torch.relu(delta))

            loss = loss_data + lambda_phy * loss_phy + lambda_mono * loss_mono
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Loss {loss_history[-1]:.6f}")

    # 6. 评估
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        preds = model(inputs).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(Y_soh[split_idx:], preds[split_idx:]))
    mae = mean_absolute_error(Y_soh[split_idx:], preds[split_idx:])

    print(f"\n测试集 RMSE: {rmse:.5f}")
    print(f"测试集 MAE:  {mae:.5f}")

    # ==========================================
    # 7. 绘图并保存 (关键修改部分)
    # ==========================================
    plt.figure(figsize=(12, 5))

    # 左图: 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Total Loss')
    plt.title("Training Loss Process")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 右图: SOH 预测对比
    plt.subplot(1, 2, 2)
    plt.plot(Y_soh, 'k-', label='True SOH', linewidth=1.5)
    plt.plot(preds, 'r--', label='PINN Prediction', linewidth=1.5)
    plt.axvline(x=split_idx, color='g', linestyle=':', label='Train/Test Split')
    plt.title(f"SOH Prediction (RMSE={rmse:.4f})")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # --- 保存图片 ---
    plt.savefig(IMAGE_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"\n[成功] 结果图片已保存至当前目录: {IMAGE_SAVE_PATH}")

    plt.show()