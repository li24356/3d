import numpy as np

import cigvis
# 2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy   shape(501,601,1101)
#RTM_P_T_32f_S1_ziti_test.npy   shape(512,1216,1920)
# Load data
a= np.load('2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy').reshape(501,601,1101)
b= np.load(r'H:\3d\outputs1\2020Z205_3D_PSTM_TIME_mini_400_2600ms\attn_light\20260116_100747\pred_mask.npy').reshape(501,601,1101)

# Create nodes

node1= cigvis.create_slices(a, cmap='seismic')

node2= cigvis.create_slices(b, cmap='gray')

# Visualize in 3D

cigvis.plot3D([node1,node2],size=(1800, 900), grid=(1, 2))