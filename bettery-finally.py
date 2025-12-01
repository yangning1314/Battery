import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns  # 引入 seaborn 画更好看的图
import os
import copy

# ==========================================
# 0. 环境配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Running on: {DEVICE}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)


# ==========================================
# 1. 数据加载与预处理 (增加验证集划分)
# ==========================================
def load_and_process_data(file_path):
    print(f"正在读取 CSV 数据: {file_path} ...")

    required_cols = [
        'soc', 'totalvoltage', 'totalcurrent',
        'minvoltagebattery', 'maxvoltagebattery',
        'mintemperaturevalue', 'maxtemperaturevalue'
    ]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in required_cols)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None

    df = df.dropna()
    df = df[(df['soc'] >= 0) & (df['soc'] <= 100)]

    target_col = 'soc'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    # 归一化 SOC 到 0-1
    y = y / 100.0

    print(f"数据总量: {len(df)}")
    return X, y


# ==========================================
# 2. 定义 MLP 模型
# ==========================================
class BatterySOC_MLP(nn.Module):
    def __init__(self, input_dim):
        super(BatterySOC_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 稍微增加 Dropout 防止过拟合

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 早停机制类 (新增)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        patience: 容忍多少个 epoch 验证集 loss 没有下降
        min_delta: 视为改进的最小变化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0


# ==========================================
# 4. 训练与评估主程序
# ==========================================
def main():
    csv_file = 'vin93.csv'
    if not os.path.exists(csv_file):
        create_dummy_csv(csv_file)

    X, y = load_and_process_data(csv_file)
    if X is None: return

    # --- 1. 科学的数据集划分 (Train/Val/Test) ---
    # 第一次划分：Train+Val (80%) 和 Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 第二次划分：Train (80% of 80% = 64%) 和 Val (20% of 80% = 16%)
    # 这里实际上是从剩余数据中分出 20% 做验证集
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    print(f"数据集划分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # --- 2. 标准化 (必须只在训练集上 fit) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # 用训练集的参数转换验证集
    X_test_scaled = scaler.transform(X_test)  # 用训练集的参数转换测试集

    # 转 Tensor
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled).to(DEVICE), torch.FloatTensor(y_train).to(DEVICE))
    val_ds = TensorDataset(torch.FloatTensor(X_val_scaled).to(DEVICE), torch.FloatTensor(y_val).to(DEVICE))
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled).to(DEVICE), torch.FloatTensor(y_test).to(DEVICE))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    # Test loader 不需要 shuffle
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # --- 3. 初始化模型组件 ---
    model = BatterySOC_MLP(input_dim=X.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # [新增] 学习率调度器: 当验证集 loss 不下降时，将学习率乘以 0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # [新增] 早停对象
    early_stopping = EarlyStopping(patience=15, min_delta=0.00001)

    # --- 4. 训练循环 (带验证) ---
    epochs = 200
    history = {'train_loss': [], 'val_loss': []}

    print("\n🚀 开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证阶段 (这是让实验有说服力的关键)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # 检查早停
        early_stopping(val_loss, model)

        if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if early_stopping.early_stop:
            print("⏹️ 早停触发！停止训练。")
            break

    # 加载表现最好的模型权重 (而不是最后一个 epoch 的权重)
    print("正在加载验证集上表现最好的模型权重...")
    model.load_state_dict(early_stopping.best_model_wts)

    # --- 5. 最终测试 (Unbiased Evaluation) ---
    print("\n正在测试集上评估最终性能...")
    model.eval()
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            y_preds.append(outputs.cpu().numpy())
            y_trues.append(targets.cpu().numpy())

    y_pred = np.vstack(y_preds)
    y_true = np.vstack(y_trues)

    # 还原百分比
    y_pred_perc = y_pred * 100
    y_true_perc = y_true * 100

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_perc, y_pred_perc))
    mae = mean_absolute_error(y_true_perc, y_pred_perc)
    r2 = r2_score(y_true_perc, y_pred_perc)

    print("=" * 40)
    print(f"🎯 最终测试集结果 (Test Set):")
    print(f"RMSE : {rmse:.4f} %")
    print(f"MAE  : {mae:.4f} %")
    print(f"R2   : {r2:.4f}")
    print("=" * 40)

    # --- 6. 高级绘图分析 ---
    plot_results(history, y_true_perc, y_pred_perc)


def plot_results(history, y_true, y_pred):
    plt.figure(figsize=(18, 5))

    # 图1: 损失曲线对比 (Train vs Val)
    # 这里的说服力在于：证明了模型没有严重的过拟合（Train和Val曲线应该贴合紧密）
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss Curve (Train vs Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图2: 回归拟合图 (Predicted vs True)
    # 这里的说服力在于：点越靠近对角线，说明预测越准
    plt.subplot(1, 3, 2)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='green')
    min_val, max_val = 0, 100
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal 1:1')
    plt.title(f'Regression: R2={r2_score(y_true, y_pred):.4f}')
    plt.xlabel('True SOC (%)')
    plt.ylabel('Predicted SOC (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图3: 误差分布直方图 (Error Histogram)
    # 这里的说服力在于：证明误差是零均值的正态分布，而不是有系统性偏差
    plt.subplot(1, 3, 3)
    errors = y_pred - y_true
    sns.histplot(errors, bins=50, kde=True, color='purple')
    plt.title('Error Distribution (Pred - True)')
    plt.xlabel('SOC Error (%)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'soc_analysis_report.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 详细分析图表已保存至: {save_path}")


# 模拟数据生成
def create_dummy_csv(filename):
    print(f"未找到 {filename}，正在生成模拟数据...")
    rows = 5000
    data = {
        'soc': np.random.uniform(0, 100, rows),
        'totalvoltage': np.random.uniform(300, 400, rows),
        'totalcurrent': np.random.normal(0, 50, rows),
        'minvoltagebattery': np.random.uniform(3.0, 4.2, rows),
        'maxvoltagebattery': np.random.uniform(3.0, 4.2, rows),
        'mintemperaturevalue': np.random.uniform(20, 40, rows),
        'maxtemperaturevalue': np.random.uniform(22, 45, rows)
    }
    pd.DataFrame(data).to_csv(filename, index=False)


if __name__ == "__main__":
    main()