import numpy as np
import cigvis
import os
from pathlib import Path

DATASETS = {
    "PSTM": {
        "data_path": Path(r"2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy"),
        "shape": (501, 601, 1101),
    },
    "RTM": {
        "data_path": Path(r"RTM_P_T_32f_S1_ziti_test.npy"),
        "shape": (512, 1216, 1920),
    },
    "F3": {
        "data_path": Path(r"F3data.npy"),
        "shape": (601, 951, 391),
    },
}

DATA_KEY = "F3"         # 选数据
cfg = DATASETS[DATA_KEY]
a = np.load(cfg["data_path"]).reshape(cfg["shape"])
b_path = Path(r'outputs999\F3data\attn_light\20260127_155059\pred_prob.npy')
b= np.load(b_path).reshape(cfg["shape"])
title_name = b_path.parents[1].name
# 必须先转为 float32 类型
b_vis = b.astype(np.float32)

# 设置阈值，将非断层部分设为 NaN (透明)
threshold = 0.1
b_vis[b_vis < threshold] = np.nan

# 3. 创建可视化节点
# 第一步：创建底图（地震数据）
nodes = cigvis.create_slices(a, cmap='seismic')


# 第二步：叠加断层 (Mask)
# 【修正点】：参数名必须是 cmaps 和 clims (复数)
nodes = cigvis.add_mask(
    nodes,
    b_vis,
    cmaps='gray_r',   # 这里改为 cmaps
    clims=[0, 1],     # 这里改为 clims
    interpolation='nearest'
)

# 4. 可视化
cigvis.plot3D(nodes, size=(1200, 900), title=title_name)