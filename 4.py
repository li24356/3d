import numpy as np
import pyvista as pv
import gc

def main():
    # --------------------------
    # 1. 数据加载与预处理 (保持原样)
    # --------------------------
    print(">>> 正在加载数据...")
    # 你的地震数据
    data_seismic = np.load('RTM_P_T_32f_S1_ziti_test.npy').reshape(512, 1216, 1920)
    
    # 你的预测数据
    data_fault_raw = np.load(r'outputs/attn_light/20251229_164230/pred_mask.npy').reshape(512, 1216, 1920)
    
    # 转为 uint8 节省内存
    data_fault = (data_fault_raw > 0.5).astype(np.uint8)
    
    # 清理内存
    del data_fault_raw
    gc.collect()

    # 检查是否有断层
    if data_fault.sum() == 0:
        print("错误：预测结果全为 0，没有检测到断层。")
        return

    # --------------------------
    # 2. 转换为 PyVista 格式
    # --------------------------
    print(">>> 正在构建 3D 场景...")
    
    # PyVista 的 wrap 可以直接把 numpy 数组包装成网格
    # 注意：PyVista 默认坐标轴顺序可能与 numpy 索引对应方式为 (X, Y, Z)
    # 这里我们直接 wrap，后续操作统一坐标系即可
    grid = pv.wrap(data_seismic)
    
    # 将断层数据添加进同一个网格，方便管理
    # 命名为 'faults'
    grid.point_data['faults'] = data_fault.ravel(order='F')  # order='F' 适配 VTK 的扁平化顺序

    # --------------------------
    # 3. 自动定位切片中心
    # --------------------------
    # 找到断层数据中的点
    indices = np.argwhere(data_fault > 0)
    # 取第一个点的坐标作为中心 (z, y, x) -> 对应 PyVista 的 (x, y, z) 还是 (z, y, x) 取决于 wrap 方式
    # 简单起见，我们取中间的一个断层点，视觉效果更好
    center_idx = len(indices) // 2
    # 注意：argwhere 返回的是 (axis0, axis1, axis2)
    # 在 pyvista wrap 后，通常 axis0 对应 X轴位置，axis1 对应 Y，axis2 对应 Z
    center_pos = indices[center_idx]
    
    print(f"切片中心锁定在: {center_pos}")

    # --------------------------
    # 4. 创建可视化对象 (Plotter)
    # --------------------------
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background('white') # 设置背景色，黑色背景容易看不清黑色断层

    # --- 图层 A: 地震数据切片 ---
    # slice_orthogonal 创建三个正交切片
    slices = grid.slice_orthogonal(x=center_pos[0], y=center_pos[1], z=center_pos[2])
    
    plotter.add_mesh(slices, 
                     scalars='values',  # 地震数据的默认名字通常是 values 或数组名
                     cmap='seismic', 
                     opacity=1.0,
                     show_scalar_bar=False) # 隐藏图例让画面更干净

    # --- 图层 B: 断层可视化 (两种方式任选) ---
    
    # 方式一：3D 实体提取 (最推荐！能看到断层的立体形状)
    # threshold 提取所有大于 0.5 的部分
    try:
        fault_body = grid.threshold(0.5, scalars='faults')
        if fault_body.n_points > 0:
            plotter.add_mesh(fault_body, 
                             color='black',       # 断层显示为黑色
                             opacity=0.6,         # 设置半透明，这样不会完全遮挡地震数据
                             label='Fault Body')
            print("已添加 3D 断层实体。")
        else:
            print("警告：阈值提取后网格为空。")
    except Exception as e:
        print(f"3D 提取出错，尝试切片模式: {e}")

    # 方式二 (备用)：断层切片叠加 (类似 cigvis 的逻辑)
    # 如果觉得 3D 实体太乱，可以用下面这行代替上面的 threshold
    # fault_slices = grid.slice_orthogonal(x=center_pos[0], y=center_pos[1], z=center_pos[2])
    # plotter.add_mesh(fault_slices, scalars='faults', cmap=['transparent', 'black'], opacity='linear')

    # --------------------------
    # 5. 显示
    # --------------------------
    plotter.add_axes() # 显示坐标轴
    plotter.show_grid() # 显示外框网格
    print(">>> 渲染完成，窗口已弹出。")
    plotter.show()

if __name__ == "__main__":
    main()
