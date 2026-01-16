import numpy as np
import torch
import segyio
import os
from scipy.ndimage import gaussian_filter
import UNet3D_Gemini  # 确保你的训练代码保存为 UNet_Gemini.py，并包含 FaultNet3D 类

# =========================================================
# 0. 配置与初始化
# =========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_FILE = '2020Z205_3D_PSTM_TIME_mini_512X512X1024.dat'
OUTPUT_SGY = 'Predicted_2020Z205_3D_FaultNet3D.sgy'
ORIGINAL_SGY = "2020Z205_3D_PSTM_TIME_mini_512X512X1024.sgy"

# 数据形状 (Inline, Crossline, Time/Depth)
# 注意：请务必确认你的数据存储顺序。通常是 (Inline, Crossline, Time)
FULL_SHAPE = (512, 512, 1024)
TILE_SIZE = (128, 128, 128)  # 3D 预测窗口大小
OVERLAP = 0.25  # 重叠率 25% (有效消除拼缝)
BATCH_SIZE = 1  # 显存不够就设为1，够大可以设为2或4


# =========================================================
# 1. 辅助函数：高斯权重生成
# =========================================================
def get_gaussian_weight(size, sigma_scale=1.0 / 8):
    """
    生成 3D 高斯权重窗口，用于消除拼接边界。
    中心权重为1，边缘权重接近0。
    """
    tmp = np.zeros(size)
    center_coords = [i // 2 for i in size]
    sigmas = [i * sigma_scale for i in size]
    tmp[tuple(center_coords)] = 1
    # 生成高斯核
    gaussian_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    # 归一化到 [0, 1]
    gaussian_map /= np.max(gaussian_map)
    return gaussian_map


# =========================================================
# 2. 核心：3D 滑动窗口预测
# =========================================================
def predict_full_volume(model, volume, tile_size, overlap):
    print("开始 3D 滑动窗口预测...")
    D, H, W = volume.shape  # 注意这里对应的是输入的维度
    d_tile, h_tile, w_tile = tile_size

    # 计算步长
    d_stride = int(d_tile * (1 - overlap))
    h_stride = int(h_tile * (1 - overlap))
    w_stride = int(w_tile * (1 - overlap))

    # 初始化累加器
    prob_map = np.zeros_like(volume, dtype=np.float32)
    count_map = np.zeros_like(volume, dtype=np.float32)

    # 获取高斯权重块
    weight_map = get_gaussian_weight(tile_size)

    # 计算总步数用于进度条
    total_steps = len(range(0, D - d_tile + d_stride, d_stride)) * \
                  len(range(0, H - h_tile + h_stride, h_stride)) * \
                  len(range(0, W - w_tile + w_stride, w_stride))
    current_step = 0

    model.eval()
    with torch.no_grad():
        # 三重循环遍历 3D 空间
        for z in range(0, D, d_stride):
            for y in range(0, H, h_stride):
                for x in range(0, W, w_stride):
                    # 1. 确定边界 (处理边缘剩余部分)
                    z_start = min(z, D - d_tile)
                    y_start = min(y, H - h_tile)
                    x_start = min(x, W - w_tile)

                    z_end = z_start + d_tile
                    y_end = y_start + h_tile
                    x_end = x_start + w_tile

                    # 2. 提取数据块
                    patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]

                    # 3. 预处理 (标准化 + 维度扩展)
                    # 必须与训练时的预处理保持一致
                    patch_tensor = torch.from_numpy(patch).float()
                    std = patch_tensor.std()
                    if std > 1e-6:
                        patch_tensor = (patch_tensor - patch_tensor.mean()) / std
                    else:
                        patch_tensor = patch_tensor - patch_tensor.mean()

                    # Clip (重要)
                    patch_tensor = torch.clamp(patch_tensor, -3.0, 3.0)

                    # Add Batch & Channel dims: [1, 1, D, H, W]
                    patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

                    # 4. 模型预测
                    # 如果使用了混合精度训练，这里也可以加上 autocast
                    outputs = model(patch_tensor)
                    probs = torch.sigmoid(outputs)

                    # 转回 CPU numpy
                    pred_patch = probs.squeeze().cpu().numpy()

                    # 5. 加权融合
                    prob_map[z_start:z_end, y_start:y_end, x_start:x_end] += pred_patch * weight_map
                    count_map[z_start:z_end, y_start:y_end, x_start:x_end] += weight_map

                    current_step += 1
                    if current_step % 50 == 0:
                        print(f"进度: {current_step}/{total_steps} blocks processed.")

    # 6. 归一化结果 (加权和 / 权重和)
    final_result = prob_map / (count_map + 1e-7)
    return final_result


# =========================================================
# 3. 主流程
# =========================================================
if __name__ == "__main__":
    # --- A. 加载模型 ---
    print(f"Loading model on {DEVICE}...")
    # 实例化新的 3D 网络
    # 注意：Base Channels 需要与训练时一致 (代码中默认是16)
    model = UNet3D_Gemini.FaultNet3D(in_channels=1, base_channels=16)

    # 加载权重
    checkpoint_path = 'FaultNet3D_Best.pth'  # 确保文件名正确
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)

    # --- B. 加载数据 ---
    print("Loading 3D seismic volume...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到数据文件: {INPUT_FILE}")

    # 读取 .dat 文件
    raw_data = np.fromfile(INPUT_FILE, dtype=np.float32)
    # 确保重塑形状正确，如果你的数据存储顺序不同，请调整这里的 reshape
    volume_3d = raw_data.reshape(FULL_SHAPE)

    print(f"Original Volume Shape: {volume_3d.shape}")

    # --- C. 执行预测 ---
    # 根据显存大小，Tile Size 可以是 (128,128,128) 或 (64,64,64)
    # Overlap 建议 0.25 (25%) 到 0.5 (50%)
    predicted_volume = predict_full_volume(
        model,
        volume_3d,
        tile_size=TILE_SIZE,
        overlap=OVERLAP
    )

    # 二值化 (可选，如果不想要概率体，只要0/1断层)
    # predicted_volume = (predicted_volume > 0.5).astype(np.float32)

    # --- E. 写入 SEGY (保持原有逻辑) ---
    print(f"Writing SEGY to {OUTPUT_SGY}...")

    # 准备写入数据：SEGY通常是一维trace的集合，需要reshape回 (TraceCount, Samples)
    # 注意：SEGY的道顺序通常对应 reshape(Trace_Count, Samples)
    # 这里假设 FULL_SHAPE 是 (Inline, Crossline, Samples) -> (512, 512, 1024)
    num_traces = FULL_SHAPE[0] * FULL_SHAPE[1]
    num_samples = FULL_SHAPE[2]

    # Flatten 前两个维度
    trace_data = predicted_volume.reshape(num_traces, num_samples)

    # 使用 segyio 复制头文件并写入数据
    if os.path.exists(ORIGINAL_SGY):
        with segyio.open(ORIGINAL_SGY, 'r', ignore_geometry=True) as src:
            spec = segyio.spec()
            spec.sorting = src.sorting
            spec.format = src.format
            spec.samples = src.samples
            spec.tracecount = src.tracecount

            with segyio.create(OUTPUT_SGY, spec) as dst:
                # 写入预测数据
                dst.trace = trace_data
                # 复制道头
                print("Copying headers...")
                for i, header in enumerate(src.header):
                    dst.header[i].update(header.items())
                    if i % 10000 == 0:
                        print(f"Header copy progress: {i}/{num_traces}")
        print("SEGY writing done.")
    else:
        print(f"Warning: Original SEGY {ORIGINAL_SGY} not found. Skipping SEGY generation.")

    print("All tasks completed.")