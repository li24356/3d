import numpy as np
import cigvis


# 2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy   shape(501,601,1101)
#RTM_P_T_32f_S1_ziti_test.npy   shape(512,1216,1920)
# Load data


# 1. 加载数据
shape = (501, 601, 1101)
a= np.load(r'2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy').reshape(501,601,1101)
b= np.load(r'outputs_final\2020Z205_3D_PSTM_TIME_mini_400_2600ms\20260116_153801\mask_thick.npy').reshape(501,601,1101)

# 2. 处理断层数据的透明度
# 必须先转为 float32 类型
b_vis = b.astype(np.float32)

# 设置阈值，将非断层部分设为 NaN (透明)
threshold = 0.5
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
cigvis.plot3D(nodes, size=(1200, 900))