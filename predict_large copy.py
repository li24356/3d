import os
from pathlib import Path
import math
import time
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import segyio
import scipy.ndimage
import torch.nn.functional as F

# ---------- 模型导入 (保持你的原始导入) ----------
from models.unet3d import UNet3D
from models.AERB3d import AERBUNet3D
from models.AERB3d import AERBUNet3DLight
from models.attention_unet3d import LightAttentionUNet3D
from models.seunet3d import SEUNet3D
from models.attention_unet3d import AttentionUNet3D

# ---------- 可编辑配置 ----------
input_path = Path(r'2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy')      # 输入文件路径
checkpoint_path = None 
checkpoints_root = Path('checkpoints3')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 维度配置
expected_shape = (501,601,1101)         # 期望的3D形状 (Z, Y, X)
expected_order = 'C'                  # 数据排列顺序：'C' 或 'F'

model_name = 'attn_light'   # 模型选择

# 模型配置
patch_size = (128, 128, 128)              # 模型训练时的输入尺寸（固定）

# [修改点1] 滑动窗口配置 - 提升至 50% 重叠以消除断层断裂
stride = (64, 64, 64)                    # 建议设为 patch_size 的一半
overlap_rate = 0.5                        # 理论重叠率

# 性能配置
batch_infer = 2                           # 适当增加Batch，因为GPU归一化效率高
use_amp = True                            # 在CUDA下启用混合精度
enable_tiled_processing = False           # 是否启用分块处理
tile_size = (256, 256, 256)               # 分块大小

# [修改点2] 归一化配置 (注意：这里主要用于记录，实际逻辑已硬编码为你的GPU Robust函数)
normalization = 'robust_gpu'              
normalize_range = (-4.0, 4.0)         

# 输出配置
out_prob_npy = Path('pred_prob.npy')      # 保存概率图（float32）
out_mask_npy = Path('pred_mask.npy')      # 保存二值化结果（uint8）
threshold = 0.5                           # 概率阈值
save_overlap_map = True                   # 保存重叠计数图

# 调试选项
debug_mode = False                        
profile_mode = False                      

# -------------------------------------------------------
#   辅助函数：高斯权重与GPU归一化
# -------------------------------------------------------

def get_gaussian_weight_map(patch_size, sigma_scale=1.0/8):
    """
    [新增] 生成3D高斯权重图，用于抑制边缘预测结果，消除拼接缝
    """
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    
    tmp[center_coords[0], center_coords[1], center_coords[2]] = 1
    weight_map = scipy.ndimage.gaussian_filter(tmp, sigmas)
    
    # 归一化到 [0, 1]
    weight_map = weight_map / weight_map.max()
    return torch.from_numpy(weight_map).float()

def _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0), device=None):
    """
    [集成] 你的训练专用归一化函数
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x)).float()
    
    if device is None:
        device = x.device
    
    x = x.to(device)
    
    if x.ndim == 5:  # (B, C, Z, Y, X)
        n_samples = x.shape[0] * x.shape[1]
        x_reshaped = x.reshape(n_samples, -1)
    elif x.ndim == 4:  # (B, Z, Y, X)
        n_samples = x.shape[0]
        x_reshaped = x.reshape(n_samples, -1)
    else:
        raise ValueError('Unsupported input ndim for normalization: %d' % x.ndim)
    
    # GPU上计算中位数
    center = torch.median(x_reshaped, dim=1, keepdim=True)[0]
    
    if use_mad:
        abs_dev = torch.abs(x_reshaped - center)
        mad = torch.median(abs_dev, dim=1, keepdim=True)[0]
        scale = torch.where(mad > 1e-6, mad * 1.4826, torch.ones_like(mad))
    else:
        scale = torch.std(x_reshaped, dim=1, keepdim=True)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    
    x_norm = (x_reshaped - center) / scale
    
    if clip_range:
        x_norm = torch.clamp(x_norm, clip_range[0], clip_range[1])
    
    if x.ndim == 5:
        x_norm = x_norm.reshape(x.shape)
    elif x.ndim == 4:
        x_norm = x_norm.reshape(x.shape)
    
    return x_norm.float()

# -------------------------------------------------------
#   通用工具函数
# -------------------------------------------------------

def print_config_summary():
    print("=" * 70)
    print("3D UNet 高斯加权滑动窗口推理 (Gaussian Weighted Inference)")
    print("=" * 70)
    print(f"输入文件: {input_path}")
    print(f"期望形状: {expected_shape}")
    print(f"Patch大小: {patch_size}")
    print(f"步长: {stride} (重叠率 ~{overlap_rate:.0%})")
    print(f"设备: {device}")
    print(f"批量大小: {batch_infer}")
    print(f"混合精度: {use_amp}")
    print(f"归一化: GPU Robust Z-Score (MAD)")
    print(f"裁剪范围: {normalize_range}")
    print("-" * 70)

def read_volume(path, expected_shape=None, expected_order='C'):
    """
    读取体积数据，支持 .npy, .npz
    自动处理二维到三维的重塑
    """
    path = Path(path)
    start_time = time.time()
    
    if path.suffix.lower() == '.npy':
        arr = np.load(path)
        load_time = time.time() - start_time
        print(f"加载完成 - 原始形状: {arr.shape}, ndim={arr.ndim}, 耗时: {load_time:.2f}秒")
        
        if arr.ndim == 2 and expected_shape is not None:
            z, y, x = expected_shape
            expected_2d_shape = (z * y, x)
            
            if arr.shape == expected_2d_shape:
                reshape_start = time.time()
                arr = arr.reshape(expected_shape, order=expected_order)
                reshape_time = time.time() - reshape_start
                print(f"重塑为3D - 新形状: {arr.shape}, 耗时: {reshape_time:.2f}秒")
            else:
                print(f"警告: 二维数组形状 {arr.shape} 与期望形状 {expected_2d_shape} 不匹配")
                raise ValueError("无法自动重塑，请检查expected_shape参数")
        
        return arr.astype(np.float32)
    
    elif path.suffix.lower() == '.npz':
        with np.load(path) as data:
            arr = data[data.files[0]]
        
        if arr.ndim == 2 and expected_shape is not None:
            z, y, x = expected_shape
            expected_2d_shape = (z * y, x)
            
            if arr.shape == expected_2d_shape:
                arr = arr.reshape(expected_shape, order=expected_order)
                print(f".npz文件重塑为3D形状: {arr.shape}")
        
        return arr.astype(np.float32)
    
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

def compute_starts(dim, psize, stride):
    if dim <= psize: return [0]
    starts = list(range(0, dim - psize + 1, stride))
    if starts[-1] + psize < dim: starts.append(dim - psize)
    return sorted(set(starts))

def calculate_patch_statistics(volume_shape, patch_size, stride):
    Z, Y, X = volume_shape
    pz, py, px = patch_size
    stz, sty, stx = stride
    starts_z = compute_starts(Z, pz, stz)
    starts_y = compute_starts(Y, py, sty)
    starts_x = compute_starts(X, px, stx)
    
    total_patches = len(starts_z) * len(starts_y) * len(starts_x)
    overlap_z = 1 - stz / pz
    overlap_y = 1 - sty / py
    overlap_x = 1 - stx / px
    
    return {
        'starts_z': starts_z, 
        'starts_y': starts_y, 
        'starts_x': starts_x,
        'total_patches': total_patches,
        'overlap_z': overlap_z,
        'overlap_y': overlap_y,
        'overlap_x': overlap_x
    }

def adjust_volume_dimensions(volume, target_shape, mode='reflect'):
    if volume.ndim != 3: raise ValueError("Volume must be 3D")
    current_shape = volume.shape
    pads = []
    slices = []
    adjustments = []
    
    for dim_idx, (c, t) in enumerate(zip(current_shape, target_shape)):
        dim_name = ['Z', 'Y', 'X'][dim_idx]
        if c < t:
            pad_before = (t - c) // 2
            pad_after = t - c - pad_before
            pads.append((pad_before, pad_after))
            slices.append(slice(None))
            adjustments.append(f"{dim_name}填充: {c}->{t}")
        elif c > t:
            start = (c - t) // 2
            end = start + t
            pads.append((0, 0))
            slices.append(slice(start, end))
            adjustments.append(f"{dim_name}裁剪: {c}->{t}")
        else:
            pads.append((0, 0))
            slices.append(slice(None))
            
    if any(p != (0, 0) for p in pads):
        volume = np.pad(volume, pads, mode=mode)
    
    if any(s != slice(None) for s in slices):
        volume = volume[tuple(slices)]
    
    if adjustments:
        print("维度调整: " + ", ".join(adjustments))
        
    return volume

def load_model_weights(model, checkpoint_path, device):
    """加载模型权重"""
    print(f"加载模型权重: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"加载checkpoint失败: {e}")
        # 尝试使用较低版本的torch加载
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, 
                                  weights_only=True)
        except:
            raise FileNotFoundError(f'无法加载checkpoint: {checkpoint_path}')
    
    possible_keys = ['model_state_dict', 'model_state', 'state_dict', 'model', 'weights']
    
    state_dict = None
    for key in possible_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            print(f"从键 '{key}' 加载权重")
            break
    
    if state_dict is None:
        state_dict = checkpoint
        print("从checkpoint直接加载权重")
    
    # 处理多GPU训练保存的模型
    if all(k.startswith('module.') for k in state_dict.keys()):
        print("移除'module.'前缀（多GPU训练模型）")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"警告: 缺失的键 ({len(missing_keys)}个): {missing_keys[:5]}" + 
              ("..." if len(missing_keys) > 5 else ""))
    if unexpected_keys:
        print(f"警告: 意外的键 ({len(unexpected_keys)}个): {unexpected_keys[:5]}" + 
              ("..." if len(unexpected_keys) > 5 else ""))
    
    if not missing_keys and not unexpected_keys:
        print("✓ 权重加载成功，所有键匹配")
    
    return model

# -------------------------------------------------------
#   核心推理逻辑 (重构版)
# -------------------------------------------------------

def sliding_inference_high_precision(volume, model, device, patch_size, stride, 
                                     batch_infer=1, use_amp=True, save_overlap=False):
    """
    高精度推理：
    1. 在Batch内部进行GPU Robust Norm，保证和训练一致。
    2. 使用高斯加权融合 (Gaussian Blending) 消除边缘。
    
    返回: (prob_map, overlap_map, coords)
    """
    Z, Y, X = volume.shape
    pz, py, px = patch_size
    
    # 1. 准备高斯权重 (GPU)
    weight_map = get_gaussian_weight_map(patch_size, sigma_scale=1.0/8).to(device)
    weight_map_cpu = None  # 延迟创建
    
    stats = calculate_patch_statistics((Z, Y, X), patch_size, stride)
    coords = [(iz, iy, ix) for iz in stats['starts_z'] 
                            for iy in stats['starts_y'] 
                            for ix in stats['starts_x']]
    
    print(f"\n推理配置:")
    print(f"  体积大小: {volume.shape}")
    print(f"  Patch大小: {patch_size}")
    print(f"  步长: {stride}")
    print(f"  总patch数: {len(coords):,}")
    print(f"  重叠率: Z={stats['overlap_z']:.1%}, Y={stats['overlap_y']:.1%}, X={stats['overlap_x']:.1%}")
    print(f"  高斯权重: sigma_scale=1/8")
    print(f"推理开始: 使用高斯加权融合...")
    
    # 2. 初始化累加器
    try:
        prob_sum = torch.zeros((Z, Y, X), device=device, dtype=torch.float32)
        count_sum = torch.zeros((Z, Y, X), device=device, dtype=torch.float32)
        on_gpu_accum = True
        print("  累加模式: GPU")
    except RuntimeError:
        print("  显存不足，切换到CPU累加模式...")
        prob_sum = torch.zeros((Z, Y, X), dtype=torch.float32)
        count_sum = torch.zeros((Z, Y, X), dtype=torch.float32)
        on_gpu_accum = False
        print("  累加模式: CPU")

    model.eval()
    
    # 性能统计
    total_time = 0
    patch_times = []
    
    with torch.no_grad():
        progress_bar = tqdm(total=len(coords), desc='推理进度', unit='patch')
        i = 0
        while i < len(coords):
            batch_start_time = time.time()
            batch_coords = coords[i:i + batch_infer]
            
            batch_tensors = []
            orig_shapes = []
            
            # --- 准备Batch数据 ---
            for (iz, iy, ix) in batch_coords:
                z_end = min(iz + pz, Z)
                y_end = min(iy + py, Y)
                x_end = min(ix + px, X)
                
                patch = volume[iz:z_end, iy:y_end, ix:x_end]
                orig_shapes.append(patch.shape)
                
                if patch.shape != (pz, py, px):
                    pad_z = pz - patch.shape[0]
                    pad_y = py - patch.shape[1]
                    pad_x = px - patch.shape[2]
                    patch = np.pad(patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
                
                batch_tensors.append(torch.from_numpy(patch))
            
            # 堆叠 -> (B, 1, Z, Y, X) -> GPU
            batch_input = torch.stack(batch_tensors).float().to(device)
            if batch_input.ndim == 4:
                batch_input = batch_input.unsqueeze(1)
                
            # --- [关键] 训练一致性归一化 ---
            batch_input = _robust_normalize_tensor_batch(
                batch_input, use_mad=True, clip_range=(-4.0, 4.0), device=device
            )
            
            # --- 模型推理 ---
            try:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(batch_input)
                        probs = torch.sigmoid(logits)
                else:
                    probs = torch.sigmoid(model(batch_input))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nGPU内存不足，尝试减小batch_infer")
                    torch.cuda.empty_cache()
                    if batch_infer > 1:
                        batch_infer = max(1, batch_infer // 2)
                        print(f"自动减小batch_infer为: {batch_infer}")
                        continue
                raise e
            
            # --- [关键] 高斯加权 ---
            probs = probs.squeeze(1) # (B, Z, Y, X)
            weighted_probs = probs * weight_map
            
            # --- 累加回大图 ---
            if not on_gpu_accum:
                weighted_probs = weighted_probs.cpu()
                if weight_map_cpu is None:
                    weight_map_cpu = weight_map.cpu()

            for bi, (iz, iy, ix) in enumerate(batch_coords):
                os = orig_shapes[bi]
                z_end, y_end, x_end = iz + os[0], iy + os[1], ix + os[2]
                
                valid_prob = weighted_probs[bi, :os[0], :os[1], :os[2]]
                
                if on_gpu_accum:
                    valid_weight = weight_map[:os[0], :os[1], :os[2]]
                    prob_sum[iz:z_end, iy:y_end, ix:x_end] += valid_prob
                    count_sum[iz:z_end, iy:y_end, ix:x_end] += valid_weight
                else:
                    valid_weight = weight_map_cpu[:os[0], :os[1], :os[2]]
                    prob_sum[iz:z_end, iy:y_end, ix:x_end] += valid_prob
                    count_sum[iz:z_end, iy:y_end, ix:x_end] += valid_weight

            # 更新进度
            batch_time = time.time() - batch_start_time
            total_time += batch_time
            patch_times.append(batch_time / len(batch_coords) if batch_coords else 0)
            
            i += len(batch_coords)
            progress_bar.update(len(batch_coords))
            
            # 更新进度条信息
            avg_time = np.mean(patch_times[-10:]) if len(patch_times) > 0 else 0
            remaining = (len(coords) - i) * avg_time if avg_time > 0 else 0
            
            progress_bar.set_postfix({
                'batch': batch_infer,
                'avg/patch': f'{avg_time:.3f}s',
                '剩余': f'{remaining/60:.1f}min'
            })
            
        progress_bar.close()
    
    print(f"\n推理统计:")
    print(f"  总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"  平均每patch: {np.mean(patch_times):.3f}秒")
    
    # --- 计算加权平均 ---
    count_sum = torch.clamp(count_sum, min=1e-6)
    final_prob = prob_sum / count_sum
    
    if on_gpu_accum:
        final_prob = final_prob.cpu()
        if save_overlap:
            count_sum = count_sum.cpu()
    
    # 检查覆盖情况
    if save_overlap:
        coverage_np = count_sum.numpy()
        uncovered = np.sum(coverage_np < 0.1)
        if uncovered > 0:
            print(f"  警告: {uncovered}个体素未被充分覆盖")
        else:
            print(f"  ✓ 所有体素都被完整覆盖")
        print(f"  覆盖统计: 最小={coverage_np.min():.2f}, 最大={coverage_np.max():.2f}, 平均={coverage_np.mean():.2f}")
            
    return final_prob.numpy(), count_sum.numpy() if save_overlap else None, coords

# -------------------------------------------------------
#   主程序
# -------------------------------------------------------

def main():
    print_config_summary()
    
    # 1. 读取数据
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
    print(f"\n1. 读取数据...")
    vol = read_volume(str(input_path), expected_shape=expected_shape, expected_order=expected_order)
    
    if vol.ndim != 3:
        raise ValueError(f"读取后不是3D数组，shape={vol.shape}")
    
    print(f"数据形状: {vol.shape}")
    print(f"数据范围: 最小值={vol.min():.6f}, 最大值={vol.max():.6f}, 均值={vol.mean():.6f}")
    
    # 2. 调整维度
    if vol.shape != expected_shape:
        print(f"\n2. 调整维度 (Reflect Padding)...")
        adjust_start = time.time()
        vol = adjust_volume_dimensions(vol, expected_shape)
        adjust_time = time.time() - adjust_start
        print(f"调整完成，耗时: {adjust_time:.2f}秒")
        print(f"最终形状: {vol.shape}")
    
    # 3. 初始化模型
    print(f"\n3. 初始化模型...")
    MODEL_REGISTRY = {
        'unet3d': UNet3D, 'aerb_light': AERBUNet3DLight, 'attn_light': LightAttentionUNet3D,
        'aerb': AERBUNet3D, 'seunet': SEUNet3D, 'attention_unet3d': AttentionUNet3D,
    }
    ModelClass = MODEL_REGISTRY.get(model_name, UNet3D)
    print(f"选择模型: {model_name} -> {ModelClass.__name__}")
    
    model = None
    for kwargs in ({'in_channels': 1, 'out_channels': 1}, {'in_channels': 1, 'base_channels': 16}, {}):
        try:
            model = ModelClass(**kwargs)
            print(f"模型初始化成功，参数: {kwargs}")
            break
        except Exception: continue
    if model is None: 
        model = UNet3D(in_channels=1, out_channels=1)
        print(f"使用默认 UNet3D")
    
    device_t = torch.device(device)
    model.to(device_t)
    print(f"模型已加载到设备: {device_t}")
    
    # 4. 智能加载 Checkpoint
    print(f"\n4. 查找模型权重...")
    ckpt_to_use = None
    def find_preferred(dir_path: Path):
        if not dir_path.exists(): return None
        for name in ('model_best_iou.pth', 'model_best.pth', f'{model_name}_best.pth'):
            if (dir_path / name).exists(): return dir_path / name
        bests = sorted(dir_path.glob('*best*.pth'), key=lambda p: p.stat().st_mtime)
        if bests: return bests[-1]
        lasts = sorted(dir_path.glob('*last*.pth'), key=lambda p: p.stat().st_mtime)
        if lasts: return lasts[-1]
        return None

    cp = Path(checkpoint_path) if checkpoint_path is not None else None
    if cp is not None and cp.exists() and cp.is_file():
        ckpt_to_use = cp
    else:
        candidate_dirs = [model_name, f'{model_name}_c16']
        for sub in candidate_dirs:
            print(f"  搜索: {checkpoints_root / sub / 'latest'}")
            ckpt_to_use = find_preferred(checkpoints_root / sub / 'latest')
            if ckpt_to_use: break
        if not ckpt_to_use:
            for sub in candidate_dirs:
                if (checkpoints_root / sub).exists():
                    subdirs = sorted([d for d in (checkpoints_root/sub).iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
                    for ts_dir in reversed(subdirs):
                        ckpt_to_use = find_preferred(ts_dir)
                        if ckpt_to_use: break
                if ckpt_to_use: break
    
    if ckpt_to_use is None:
        raise FileNotFoundError(f'未找到 {model_name} 的checkpoint')
    
    final_ckpt = Path(ckpt_to_use)
    print(f"✓ 找到权重: {final_ckpt}")
    model = load_model_weights(model, final_ckpt, device_t)
    
    # 5. 推理
    print(f"\n5. 开始高精度滑动窗口推理 (Gaussian Weighted)...")
    print("注意: 归一化将在GPU的Patch层级进行，以匹配训练逻辑")
    
    inference_start = time.time()
    prob_map, overlap_map, coords = sliding_inference_high_precision(
        volume=vol,
        model=model,
        device=device_t,
        patch_size=patch_size,
        stride=stride,
        batch_infer=batch_infer,
        use_amp=use_amp,
        save_overlap=save_overlap_map
    )
    inference_time = time.time() - inference_start
    
    # 6. 分析结果
    print(f"\n6. 结果分析...")
    print(f"概率图范围: 最小值={prob_map.min():.6f}, 最大值={prob_map.max():.6f}")
    print(f"概率图统计: 均值={prob_map.mean():.6f}, 标准差={prob_map.std():.6f}")
    
    # 二值化
    mask = (prob_map >= threshold).astype(np.uint8)
    pos_ratio = mask.mean()
    print(f"二值化统计: 阈值={threshold}, 正样本比例={pos_ratio:.4%}")
    print(f"             正样本数={mask.sum():,}, 负样本数={mask.size - mask.sum():,}")
    
    # 7. 保存结果
    print(f"\n7. 保存结果...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_name = Path(input_path).stem
    output_dir = Path('outputs1') / input_name / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 二值化
    mask = (prob_map >= threshold).astype(np.uint8)
    pos_ratio = mask.mean()
    
    # 只保存二值mask（不保存概率图）
    mask_path = output_dir / out_mask_npy.name
    np.save(mask_path, mask)
    print(f"✓ 二值mask保存到: {mask_path}")
    
    # 不保存重叠图
    
    # 保存处理摘要
    summary = {
        'model': model_name,
        'checkpoint': str(final_ckpt),
        'timestamp': timestamp,
        'input_file': str(input_path),
        'input_shape': list(vol.shape),
        'patch_size': list(patch_size),
        'stride': list(stride),
        'overlap_rate': overlap_rate,
        'normalization': normalization,
        'threshold': float(threshold),
        'inference_time_s': float(inference_time),
        'inference_time_min': float(inference_time / 60),
        'positive_ratio': float(pos_ratio),
        'output_dir': str(output_dir)
    }
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ 处理摘要保存到: {summary_path}")
    
    # 8. 生成 SEGY
    print(f"\n8. 生成 SEG-Y 文件...")
    ORIGINAL_FILE = r'2020Z205_3D_PSTM_TIME_mini_400_2600ms.sgy'
    PREDICTED_NPY = mask_path
    OUTPUT_FILE = output_dir / 'predicted_result_segyio_final1.sgy'
    
    if os.path.exists(ORIGINAL_FILE) and PREDICTED_NPY.exists():
        try:
            print(f"  原始 SGY: {ORIGINAL_FILE}")
            print(f"  预测 NPY: {PREDICTED_NPY}")
            
            predicted_data_np = np.load(PREDICTED_NPY).astype(np.float32)
            
            # 验证维度
            if predicted_data_np.ndim != 3:
                raise ValueError(f"期望3D数据，实际维度: {predicted_data_np.ndim}")
            
            if predicted_data_np.shape != expected_shape:
                print(f"  警告: 预测形状 {predicted_data_np.shape} != 期望 {expected_shape}")
            
            N_samples = predicted_data_np.shape[2]
            predicted_data_2d = predicted_data_np.reshape(-1, N_samples)
            N_traces = predicted_data_2d.shape[0]
            
            print(f"  -> 数据已重塑: ({N_traces} 道, {N_samples} 采样点)")
            
            with segyio.open(ORIGINAL_FILE, ignore_geometry=True) as src:
                spec = segyio.spec()
                spec.ilines = src.ilines
                spec.xlines = src.xlines
                spec.samples = src.samples
                spec.format = 5
                spec.tracecount = N_traces
                
                original_tracecount = src.tracecount
                if original_tracecount != N_traces:
                    print(f"  警告：原始文件道数 ({original_tracecount}) 与预测 ({N_traces}) 不匹配")
                
                N_copy_traces = min(original_tracecount, N_traces)
                
                print(f"  正在写入: {OUTPUT_FILE} ...")
                with segyio.create(str(OUTPUT_FILE), spec) as dst:
                    dst.text[0] = src.text[0]
                    dst.bin = src.bin
                    for i in tqdm(range(N_traces), desc='  写入SEG-Y', unit='trace'):
                        dst.trace[i] = predicted_data_2d[i]
                        if i < N_copy_traces:
                            dst.header[i] = src.header[i]
                        else:
                            dst.header[i] = src.header[0] if original_tracecount > 0 else {}
                            dst.header[i][segyio.cdp] = i + 1
                            dst.header[i][segyio.traceno] = i + 1
                            
            print(f"✓ SEG-Y 文件生成成功: {OUTPUT_FILE}")
        except Exception as e:
            print(f"SEGY生成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not os.path.exists(ORIGINAL_FILE):
            print(f"  错误：原始 SEG-Y 文件未找到：{ORIGINAL_FILE}")
        print("  -> 跳过 SEG-Y 生成。")

    print(f"\n所有输出已保存到目录: {output_dir}")
    
    print("\n" + "=" * 70)
    print("高精度推理完成!")
    print("=" * 70)
    print(f"关键信息:")
    print(f"  • 模型: {model_name}")
    print(f"  • 总patch数: {len(coords):,}")
    print(f"  • 推理时间: {inference_time/60:.1f} 分钟")
    print(f"  • 正样本比例: {pos_ratio:.4%}")
    print(f"  • 输出文件:")
    print(f"      - {out_mask_npy.name} (二值分割)")
    print(f"      - inference_summary.json (处理摘要)")
    print(f"      - predicted_result_segyio_final1.sgy (SEG-Y格式)")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
    except Exception as e:
        print(f"\n错误发生: {e}")
        import traceback
        traceback.print_exc()