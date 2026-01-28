import numpy as np

import cigvis
# 2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy   shape(501,601,1101)
#RTM_P_T_32f_S1_ziti_test.npy   shape(512,1216,1920)
# F3data.npy  shape (601,951,391) 



# Load data (.dat 为二进制浮点文件，使用 fromfile 读取)
a = np.load(r'F3data.npy').reshape(601,951,391) 

b = np.fromfile(r'train\fault\0.dat', dtype=np.float32).reshape(128, 128, 128)
print(b.shape)

# Create nodes

node1= cigvis.create_slices(a, cmap='seismic')

node2= cigvis.create_slices(b, cmap='gray')

# Visualize in 3D

cigvis.plot3D([node1,node2],size=(1800, 900), grid=(1, 2))
