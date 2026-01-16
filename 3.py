import numpy as np
import cigvis
from matplotlib.colors import ListedColormap
import gc  # 引入垃圾回收模块，用于手动释放内存

def main():
    # 1. 加载您的真实数据
    print("正在从 .npy 文件加载您的真实数据...")
    
    # 假设您的数据是 float32，占用约 4.6GB 内存
    # 建议：如果数据本身就是 3D 的，尽量避免硬编码 reshape，这里保留您的写法
    a = np.load('RTM_P_T_32f_S1_ziti_test.npy').reshape(512, 1216, 1920)
    
    # 加载预测结果
    b = np.load(r'outputs/attn_light/20251229_164230/pred_mask.npy').reshape(512, 1216, 1920)
    print("数据加载完成。")

    # 2. 对断层数据应用阈值并优化内存
    print("正在对断层数据应用阈值...")
    
    # 优化点：使用 uint8 类型代替 float32。
    # 0 或 1 的数据用 uint8 足够，内存占用将从 4.6GB 降至约 1.1GB
    b_binary = (b > 0.5).astype(np.uint8)
    
    print("阈值应用完成，正在清理原始预测数据以释放内存...")
    
    # 优化点：b_binary 生成后，原始的 b 就不再需要了，手动删除并回收
    del b
    gc.collect()
    
    print("内存清理完成。")

    # 统计断层点数量
    fault_point_count = np.sum(b_binary)

    if fault_point_count == 0:
        print("警告：在应用阈值后，没有找到任何断层点。程序终止。")
        return

    # --- 这是解决问题的关键步骤 ---
    # 3. 自动查找一个断层点的位置，以确保切片能“看到”它
    print("正在查找一个断层点的坐标...")
    
    # np.where 返回的是一个 tuple (array_z, array_y, array_x)
    fault_coords = np.where(b_binary > 0)
    
    # 获取第一个找到的断层点的三维坐标
    # 注意：这里的索引顺序取决于 numpy 数组的存储顺序 (dim0, dim1, dim2)
    # 通常对应地震数据的 (Inline, Crossline, Time/Depth)
    y_fault = fault_coords[0][0]
    x_fault = fault_coords[1][0]
    z_fault = fault_coords[2][0]
    
    print(f"成功找到一个断层点，其坐标为 (Dim0/Inline={y_fault}, Dim1/Xline={x_fault}, Dim2/Time={z_fault})。")
    print("将使用此坐标作为切片的中心位置。")

    # 4. 创建可视化节点，并指定切片位置
    # 在 cigvis 中, iline 对应第一轴(dim0), xline 对应第二轴(dim1), tslice 对应第三轴(dim2)
    print("正在创建以断层点为中心的切片节点...")
    
    # 节点1：背景地震数据
    node1 = cigvis.create_slices(a, iline=y_fault, xline=x_fault, tslice=z_fault, cmap='seismic')
    
    # 节点2：断层数据（叠加层）
    fault_colors = [(0, 0, 0, 0),  # 0值：透明
                    (0, 0, 0, 1)]  # 1值：黑色
    fault_cmap = ListedColormap(fault_colors)
    
    # 创建断层切片
    # interpolation='nearest' 可以防止二值图像边缘模糊（如果 cigvis 版本支持该参数，推荐加上）
    # 如果报错提示不支持 interpolation 参数，请删除该参数
    node2 = cigvis.create_slices(b_binary, iline=y_fault, xline=x_fault, tslice=z_fault, 
                                 cmap=fault_cmap, clim=[0, 1], interpolation='nearest')

    print("节点创建完成。")

    # 5. 将两个节点在同一个3D视图中绘制
    print("正在生成三维叠加图...")
    
    cigvis.plot3D([node1, node2],
                  title='地震数据(a)与断层(b)的叠加显示',
                  xlabel='Xline (Dim1)',
                  ylabel='Inline (Dim0)',
                  zlabel='Time/Depth (Dim2)')
                  
    print("图像生成完成。")

if __name__ == "__main__":
    main()
