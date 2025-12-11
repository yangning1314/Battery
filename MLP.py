import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging

# ==========================================
# 0. 基础配置
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_FOLDER = 'Batch-1'
SAVE_IMG_PATH = 'Final_Comparison.png'


# 设置随机种子保证复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(1234)


# Logger
def get_logger():
    logger = logging.getLogger('PINN_Final')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    return logger


logger = get_logger()


class AverageMeter(object):
    def __init__(self): self.reset()

    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1):
        self.val = val;
        self.sum += val * n;
        self.count += n;
        self.avg = self.sum / self.count


# ==========================================
# 1. 模型定义 (保持原样)
# ==========================================
class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, layers_num=4, hidden_dim=64, droupout=0.0):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(Sin())
                if droupout > 0: self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear): nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class Solution_u(nn.Module):
    def __init__(self, input_dim):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=input_dim, output_dim=32, layers_num=3, hidden_dim=64)
        self.predictor = nn.Sequential(
            nn.Linear(32, 32),
            Sin(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.predictor(x)


class PINN(nn.Module):
    def __init__(self, feature_dim, alpha=0.05, beta=0.05):
        super(PINN, self).__init__()
        self.solution_u = Solution_u(input_dim=feature_dim).to(DEVICE)

        # 物理网络
        dim_F = feature_dim + 1 + (feature_dim - 1) + 1
        self.dynamical_F = MLP(input_dim=dim_F, output_dim=1, layers_num=3, hidden_dim=64).to(DEVICE)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        self.alpha = alpha  # PDE 权重
        self.beta = beta  # 单调性权重

    def forward(self, xt):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]

        u = self.solution_u(xt)

        # 计算导数
        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True)[0]

        F_input = torch.cat([xt, u, u_x, u_t], dim=1)
        F = self.dynamical_F(F_input)

        f = u_t - F
        return u, f


# ==========================================
# 2. 数据处理
# ==========================================
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X) - 1

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx + 1], self.y[idx], self.y[idx + 1]


def load_data(folder):
    train_ids = [1, 2, 3, 4, 5, 6]
    test_ids = [7, 8]  # 测试集

    def read_ids(ids):
        xs, ys = [], []
        for bid in ids:
            path = os.path.join(folder, f"2C_battery-{bid}.csv")
            try:
                df = pd.read_csv(path)
                if df.shape[1] < 2: df = pd.read_csv(path, sep='\t')
                df.replace([np.inf, -np.inf], np.nan, inplace=True);
                df.dropna(inplace=True)

                feat = df.iloc[:, :-1].values
                soh = df.iloc[:, -1].values / 2.0

                # 添加时间列 t
                t = np.linspace(0, 1, len(feat)).reshape(-1, 1)
                x_combined = np.hstack((feat, t))

                xs.append(x_combined)
                ys.append(soh)
            except:
                pass
        return np.concatenate(xs), np.concatenate(ys)

    X_train, y_train = read_ids(train_ids)
    X_test, y_test = read_ids(test_ids)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# ==========================================
# 3. 改进后的训练流程 (Warm-up Strategy)
# ==========================================

def train_baseline_mlp(X_train, y_train, input_dim, epochs=1000):
    logger.info(f"\n>>> Training MLP (Baseline)...")
    model = MLP(input_dim=input_dim, output_dim=1, layers_num=4, hidden_dim=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

    dataset = BatteryDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x1, _, y1, _ in loader:
            x1, y1 = x1.to(DEVICE), y1.to(DEVICE)
            pred = model(x1)
            loss = nn.MSELoss()(pred, y1)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
        scheduler.step()
    return model


def train_pinn_with_warmup(X_train, y_train, input_dim, epochs=1000, warmup=200):
    """
    关键修改：加入 warmup 阶段
    - 前 warmup 轮：alpha=0, beta=0 (只像 MLP 一样学数据)
    - 后续：加入物理约束进行微调
    """
    logger.info(f"\n>>> Training PINN (With Warm-up Strategy)...")

    # 注意：这里把 alpha/beta 调小了一点，防止物理约束过强导致欠拟合
    model = PINN(feature_dim=input_dim, alpha=0.01, beta=0.05).to(DEVICE)

    opt1 = optim.Adam(model.solution_u.parameters(), lr=0.005)  # 主网络
    opt2 = optim.Adam(model.dynamical_F.parameters(), lr=0.005)  # 物理网络

    sch1 = optim.lr_scheduler.StepLR(opt1, step_size=400, gamma=0.5)
    sch2 = optim.lr_scheduler.StepLR(opt2, step_size=400, gamma=0.5)

    dataset = BatteryDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(epochs):
        model.train()
        l1_m = AverageMeter();
        l2_m = AverageMeter();
        l3_m = AverageMeter()

        # 动态调整物理权重：Warmup 阶段为 0
        current_alpha = 0.0 if epoch < warmup else model.alpha
        current_beta = 0.0 if epoch < warmup else model.beta

        for x1, x2, y1, y2 in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            y1, y2 = y1.to(DEVICE), y2.to(DEVICE)

            u1, f1 = model(x1)
            u2, f2 = model(x2)

            # 1. Data Loss
            l_data = 0.5 * nn.MSELoss()(u1, y1) + 0.5 * nn.MSELoss()(u2, y2)

            # 2. PDE Loss (Warmup 时忽略)
            l_pde = torch.tensor(0.0).to(DEVICE)
            if current_alpha > 0:
                f_target = torch.zeros_like(f1)
                l_pde = 0.5 * nn.MSELoss()(f1, f_target) + 0.5 * nn.MSELoss()(f2, f_target)

            # 3. Mono Loss (Warmup 时忽略)
            l_phy = torch.tensor(0.0).to(DEVICE)
            if current_beta > 0:
                # 约束：预测的衰减方向应该和真实数据一致，且应该平滑
                # 如果真实数据衰减(y1>y2)，预测必须衰减(u1>u2)
                l_phy = model.relu((u2 - u1) * (y1 - y2)).sum()

            total_loss = l_data + current_alpha * l_pde + current_beta * l_phy

            opt1.zero_grad();
            opt2.zero_grad()
            total_loss.backward()
            opt1.step();
            opt2.step()

            l1_m.update(l_data.item());
            l2_m.update(l_pde.item());
            l3_m.update(l_phy.item())

        sch1.step();
        sch2.step()

        if (epoch + 1) % 100 == 0:
            status = "Warmup" if epoch < warmup else "Physics-Informed"
            logger.info(
                f"[PINN] Epoch {epoch + 1} ({status}): Data {l1_m.avg:.5f} | PDE {l2_m.avg:.5f} | Mono {l3_m.avg:.5f}")

    return model


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # A. 数据
    X_train, y_train, X_test, y_test = load_data(DATA_FOLDER)
    feat_dim = X_train.shape[1]
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # B. 训练 (对比 1000 轮)
    mlp_model = train_baseline_mlp(X_train, y_train, feat_dim, epochs=1000)
    pinn_model = train_pinn_with_warmup(X_train, y_train, feat_dim, epochs=1000, warmup=200)

    # C. 评估
    mlp_model.eval();
    pinn_model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pred_mlp = mlp_model(inputs).cpu().numpy()
        pred_pinn = pinn_model.solution_u(inputs).cpu().numpy()

    rmse_mlp = np.sqrt(mean_squared_error(y_test, pred_mlp))
    rmse_pinn = np.sqrt(mean_squared_error(y_test, pred_pinn))

    # 计算提升百分比
    imp = (rmse_mlp - rmse_pinn) / rmse_mlp * 100

    logger.info("\n" + "=" * 40)
    logger.info(f"FINAL RESULT COMPARISON")
    logger.info("=" * 40)
    logger.info(f"MLP  RMSE: {rmse_mlp:.5f}")
    logger.info(f"PINN RMSE: {rmse_pinn:.5f}")
    logger.info(f"Improvement: {imp:.2f}%")
    logger.info("=" * 40)

    # D. 绘图
    plt.figure(figsize=(12, 6))

    # 为了展示效果，我们只画 Test Set 中后半段的数据（对应 Battery 8）
    # 这样曲线是连续的，不会因为 Battery 7 跳到 Battery 8 而断裂
    start_idx = int(len(y_test) * 0.5)
    cycles = np.arange(len(y_test[start_idx:]))

    plt.plot(cycles, y_test[start_idx:], 'k-', label='True SOH', linewidth=2, alpha=0.5)
    plt.plot(cycles, pred_mlp[start_idx:], 'b:', label=f'MLP ({rmse_mlp:.4f})', linewidth=1.5)
    plt.plot(cycles, pred_pinn[start_idx:], 'r-', label=f'PINN ({rmse_pinn:.4f})', linewidth=2)  # 实线强调

    plt.title(f'Generalization on Unseen Battery (Imp: {imp:.2f}%)')
    plt.xlabel('Cycle')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(SAVE_IMG_PATH, dpi=300, bbox_inches='tight')
    logger.info(f"Result saved to {SAVE_IMG_PATH}")
    plt.show()