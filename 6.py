import numpy as np
import matplotlib.pyplot as plt
import os

# 修改这里的路径为你新生成的文件夹路径
data_root = './synthetic_data_v2/train' 

# 读取第 0 个样本
seis = np.load(os.path.join(data_root, 'seismic', '5.npy'))
label = np.load(os.path.join(data_root, 'label', '5.npy'))

print(f"数据形状: {seis.shape}, 标签形状: {label.shape}")
print(f"数据范围: min={seis.min():.2f}, max={seis.max():.2f}")
print(f"标签值: {np.unique(label)}") # 应该只有 0 和 1

# 可视化 (Inline 切片)
idx = 50
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("New Synthetic Seismic")
plt.imshow(seis[:, idx, :], cmap='gray', aspect='auto') # 垂直轴是时间
plt.colorbar()

plt.subplot(122)
plt.title("New Label")
plt.imshow(label[:, idx, :], cmap='jet', aspect='auto', interpolation='nearest')
plt.colorbar()

plt.show()