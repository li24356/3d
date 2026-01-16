import numpy as np
import segyio
import os

# from obspy import read # 暂时不需要，因为您主要使用 segyio 处理结构和头信息

# --- 文件路径 ---
# 请确保这些路径正确，否则会引发 FileNotFoundError
ORIGINAL_FILE = r'2020Z205_3D_PSTM_TIME_mini_400_2600ms.sgy'
PREDICTED_NPY = r"outputs\attn_light\20251218_204905\pred_mask.npy"

# 将输出文件保存到与 PREDICTED_NPY 相同的目录
import os
pred_dir = os.path.dirname(PREDICTED_NPY) or '.'
OUTPUT_FILE = os.path.join(pred_dir, 'predicted_result_segyio_final1.sgy')

# --- 核心转换逻辑 ---

# 1.1 检查文件存在性
if not os.path.exists(ORIGINAL_FILE):
    print(f"错误：原始 SEG-Y 文件未找到：{ORIGINAL_FILE}")
    exit()
if not os.path.exists(PREDICTED_NPY):
    print(f"错误：预测 NumPy 文件未找到：{PREDICTED_NPY}")
    exit()

try:
    # --- 步骤 1: 准备数据和获取原始结构 ---

    # 1.2 加载预测数据并重塑为 [traces, samples]
    print(f"1. 读取预测数据: {PREDICTED_NPY}...")
    # 注意：这里假设您的预测数据是一个三维数组 (IL, XL, T)
    predicted_data_np = np.load(PREDICTED_NPY).astype(np.float32)
    # 将 IL/XL/T 重塑为 (TraceCount, N_samples)
    N_samples = predicted_data_np.shape[2]
    predicted_data_2d = predicted_data_np.reshape(-1, N_samples)
    
    N_traces, _ = predicted_data_2d.shape
    print(f"   - 预测数据形状已重塑为: ({N_traces} 道, {N_samples} 采样点)")

    # 1.3 使用 segyio 读取原始文件，获取其结构（文件头和道头模板）
    print(f"2. 读取原始 SEG-Y 文件结构: {ORIGINAL_FILE}...")
    with segyio.open(ORIGINAL_FILE, ignore_geometry=True) as src:
        
        # 获取原始文件的配置信息
        spec = segyio.spec()
        spec.ilines = src.ilines
        spec.xlines = src.xlines
        spec.samples = src.samples
        spec.format = 5 # 4-byte IEEE float (标准浮点格式)
        
        # *** 修正错误：使用 0 表示 NotSorted (未排序) ***
        # segyio.Sort 不存在，0 是 SEG-Y 规范中 NotSorted 的标准值
        spec.sorting = 0 
        # **********************************************
        
        spec.tracecount = N_traces # 新文件的道数以预测数据为准
        
        # 检查道数是否匹配
        original_tracecount = src.tracecount
        if original_tracecount != N_traces:
            print(f"   - 警告：原始文件道数 ({original_tracecount}) 与预测数据道数 ({N_traces}) 不匹配。")
        
        N_copy_traces = min(original_tracecount, N_traces)

        # --- 步骤 2: 写入新文件 ---
        print(f"3. 创建并写入新文件: {OUTPUT_FILE}...")
        with segyio.create(OUTPUT_FILE, spec) as dst:
            
            # 2.1 复制文件头信息 (文本头和二进制头)
            # segyio 会自动初始化新的头，但复制能确保所有元数据继承
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            
            # 2.2 复制道头和写入数据
            for i in range(N_traces):
                # 写入数据
                dst.trace[i] = predicted_data_2d[i]
                
                # 复制道头：只复制原始文件存在的道头
                if i < N_copy_traces:
                    # 批量复制所有道头字段
                    dst.header[i] = src.header[i]
                else:
                    # 如果预测道数比原始道数多，我们复制第一个道头作为模板，并更新关键字段
                    if original_tracecount > 0:
                        dst.header[i] = src.header[0] # 复制第一个头作为模板
                        # 确保更新道号 (TRACENO) 和 CDP/X-LINE/IN-LINE
                        dst.header[i][segyio.cdp] = i + 1
                        dst.header[i][segyio.traceno] = i + 1
                        # 注意：对于新增的道，无法准确复制 ILINE/XLINE 几何信息
                    else:
                        # 如果原始文件是空的，至少设置道号
                        dst.header[i][segyio.cdp] = i + 1
                        dst.header[i][segyio.traceno] = i + 1
                        
            print(f"4. 预测文件已成功转换为 {OUTPUT_FILE}，共 {N_traces} 道。")
            print("-" * 30)

except Exception as e:
    print("-" * 30)
    print(f"使用 segyio 转换时发生严重错误: {e}")
    print("请检查：\n1. 文件路径是否正确。\n2. 原始 SEG-Y 文件是否损坏。\n3. NumPy 数据的形状是否与原始文件道数和采样点匹配。")