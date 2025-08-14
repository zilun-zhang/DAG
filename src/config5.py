# config5.py
from pathlib import Path
import torch

# === Paths ===
DATA_DIR = Path(r"E:/PythonProject6/data/gpickle1")
CHECKPOINT_DIR = Path("checkpoints5")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# === Training params ===
EPOCHS = 700
LR = 1e-3
WEIGHT_DECAY = 1e-5
PRINT_EVERY = 1
SAVE_EVERY = 0

# === Structure constraints ===
MAX_LONGEST_PATH_TIME = 300.0
EDGE_TIME_MIN = 0.05

# === Normalization ===
NORM_N = 100.0
NORM_E = 400.0
NORM_L = 10.0
NORM_W = 10.0
NORM_T = 300.0


# === New: loss weights (三层：损失层) ===
W_BCE = 1.0
W_TIME = 0.2
W_TOTALT = 0.05
W_LONGEST = 0.05
W_NODE_TIME_UNI = 0.1    # 新增：节点出边时间一致性
W_DAG = 0.0
W_DEG_COV = 0.05       # 度覆盖（强烈建议开）
W_SRC_SINK_SOFT = 0.02 # 软的单源/单汇
W_TIME = 0.2       # 边的时间 L1，仅在有边处比较
W_TOTALT = 0.05    # 让全图总时间接近目标
W_LONGEST = 0.05   # 约束最长路径时间
W_TIME_NODE = 0.50      # ★ 节点时间损失（主力）

# 软单源/单汇的平滑参数
SRC_SINK_TAU = 0.1     # “零度阈值”附近
SRC_SINK_K   = 10.0    # 平滑斜率（越大越趋近于硬阈）

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
