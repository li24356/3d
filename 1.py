import segyio
import numpy as np

def sgy_to_npy(sgy_path, npy_path):
    """
    读取sgy文件，去除卷头和道头，仅保存地震数据为numpy格式。
    """
    try:
        # ignore_geometry=True 强制按“道集”读取，不解析3D线号，避免因头文件信息缺失导致的报错
        with segyio.open(sgy_path, mode='r', ignore_geometry=True) as f:
            
            print(f"正在读取文件: {sgy_path}")
            
            # --- 修改部分：使用 len(f.trace) 获取道数 ---
            n_traces = len(f.trace)
            n_samples = len(f.samples)
            
            print(f"道数 (Traces): {n_traces}")
            print(f"采样点数 (Samples): {n_samples}")
            
            # 读取所有道数据
            # f.trace.raw[:] 会自动去除所有卷头和道头，只提取纯数据
            data = f.trace.raw[:]
            
            # 确保数据是 numpy 数组（通常默认就是）
            # 如果需要特定的浮点精度（例如 float32），可以在这里转换
            # data = np.asarray(data, dtype=np.float32)

            print(f"数据读取完成，形状为: {data.shape}")
            
            # 保存
            np.save(npy_path, data)
            print(f"数据已成功保存为: {npy_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    # 请确保文件名与你实际的文件名一致
    input_file = 'RTM_P_T_32f_S1_ziti_test.sgy' 
    output_file = 'RTM_P_T_32f_S1_ziti_test.npy'
    
    sgy_to_npy(input_file, output_file)
