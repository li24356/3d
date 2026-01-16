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
from models.unet3d import UNet3D
from models.AERB3d import AERBUNet3D
from models.unet3d import UNet3D
from models.AERB3d import AERBUNet3DLight
from models.attention_unet3d import LightAttentionUNet3D
from models.seunet3d import SEUNet3D
from models.attention_unet3d import AttentionUNet3D
# ---------- 可编辑配置（直接在文件中修改） ----------
input_path = Path(r'2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy')      # 输入文件路径
# checkpoint_path = Path('checkpoints/unet3d_best.pth')
checkpoint_path = None 
checkpoints_root = Path('checkpoints3')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 新增：选择要测试/推理的模型 key（与下方 MODEL_REGISTRY 对应）

# 维度配置
expected_shape = (501,601,1101)         # 期望的3D形状 (Z, Y, X)
expected_order = 'C'                      # 数据排列顺序：'C' 或 'F'


model_name = 'attention_unet3d'   # 可改为 'attn_light' / 'aerb_light' / 你的自定义 key/ 'unet3d' 



# 模型配置
patch_size = (128, 128, 128)              # 模型训练时的输入尺寸（固定）

# 滑动窗口配置 - 高精度模式（25%重叠率）
stride = (96, 96, 96)                    # 25%重叠率，最高精度
overlap_rate = 0.25                        # 重叠比例

# 性能配置
batch_infer = 1                           # 一次送入GPU的patch数（显存小设为1）
use_amp = True                            # 在CUDA下启用混合精度
enable_tiled_processing = False           # 是否启用分块处理（超大数据时使用）
tile_size = (256, 256, 256)               # 分块大小

# 数据预处理
normalization = 'robust'               # 改为和 evaluate.py 一致的鲁棒Z-Score
normalize_range = (-1.0, 1.0)         # 保留但在 robust/zscore 下不使用

# 输出配置
out_prob_npy = Path('pred_prob.npy')      # 保存概率图（float32）
out_mask_npy = Path('pred_mask.npy')      # 保存二值化结果（uint8）
threshold = 0.5                           # 概率阈值
save_overlap_map = True                   # 保存重叠计数图（用于验证覆盖）


# 调试选项
debug_mode = False                        # 调试模式，输出更多信息
profile_mode = False                      # 性能分析模式
# -------------------------------------------------------

def print_config_summary():
    """打印配置摘要"""
    print("=" * 70)
    print("3D UNet 滑动窗口推理 - 高精度模式配置")
    print("=" * 70)
    print(f"输入文件: {input_path}")
    print(f"期望形状: {expected_shape}")
    print(f"Patch大小: {patch_size}")
    print(f"步长: {stride}")
    print(f"重叠率: {overlap_rate:.1%}")
    print(f"设备: {device}")
    print(f"批量大小: {batch_infer}")
    print(f"混合精度: {use_amp}")
    print(f"归一化: {normalization} {normalize_range if normalization=='minmax' else ''}")
    print("-" * 70)

def read_volume(path, expected_shape=None, expected_order='C'):
    """
    读取体积数据，支持 .npy, .npz, .dat
    自动处理二维到三维的重塑
    """
    path = Path(path)
    start_time = time.time()
    
    if path.suffix.lower() == '.dat':
        raise ValueError("请使用 .npy 或 .npz 格式，或单独处理 .dat 文件")
    
    elif path.suffix.lower() == '.npy':
        arr = np.load(path)
        load_time = time.time() - start_time
        print(f"加载完成 - 原始形状: {arr.shape}, ndim={arr.ndim}, 耗时: {load_time:.2f}秒")
        
        # 如果是二维数组且提供了期望形状，尝试重塑
        if arr.ndim == 2 and expected_shape is not None:
            z, y, x = expected_shape
            expected_2d_shape = (z * y, x)
            
            if arr.shape == expected_2d_shape:
                # 重塑为3D
                reshape_start = time.time()
                arr = arr.reshape(expected_shape, order=expected_order)
                reshape_time = time.time() - reshape_start
                print(f"重塑为3D - 新形状: {arr.shape}, 耗时: {reshape_time:.2f}秒")
            else:
                print(f"警告: 二维数组形状 {arr.shape} 与期望形状 {expected_2d_shape} 不匹配")
                # 尝试直接计算合适的形状
                total_elements = arr.size
                # 寻找最接近期望形状的因子分解
                factors = []
                for i in range(1, int(math.sqrt(total_elements)) + 1):
                    if total_elements % i == 0:
                        factors.append((i, total_elements // i))
                
                if factors:
                    # 选择最接近正方形的一组
                    best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                    print(f"建议形状: {best_factor} 或 ({best_factor[1]}, {best_factor[0]})")
                raise ValueError("无法自动重塑，请检查expected_shape参数")
        
        return arr.astype(np.float32)
    
    elif path.suffix.lower() == '.npz':
        with np.load(path) as data:
            # 获取第一个数组
            if len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                # 尝试常见的键名
                for key in ['data', 'volume', 'image', 'arr_0']:
                    if key in data:
                        arr = data[key]
                        break
                else:
                    arr = data[data.files[0]]
        
        # 同样处理二维到三维的重塑
        if arr.ndim == 2 and expected_shape is not None:
            z, y, x = expected_shape
            expected_2d_shape = (z * y, x)
            
            if arr.shape == expected_2d_shape:
                arr = arr.reshape(expected_shape, order=expected_order)
                print(f".npz文件重塑为3D形状: {arr.shape}")
        
        return arr.astype(np.float32)
    
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

def normalize_patch(patch, method='zscore', norm_range=(-1.0, 1.0), eps=1e-8):
    """
    与评估端一致的归一化：
    - method='robust': 中位数 + MAD（1.4826系数） + 裁剪到 [-4, 4]
    - method='zscore': 均值/标准差
    支持输入形状: (Z,Y,X)、(B,Z,Y,X)、(B,1,Z,Y,X)
    """
    patch = patch.astype(np.float32)

    # 统一到 (B, C, Z, Y, X)
    if patch.ndim == 5 and patch.shape[1] == 1:
        data = patch
    elif patch.ndim == 4:
        data = patch[:, None, ...]
    elif patch.ndim == 3:
        data = patch[None, None, ...]
    else:
        raise ValueError(f"不支持的 patch 维度: {patch.shape}")

    if method == 'robust':
        B, C, Z, Y, X = data.shape
        flat = data.reshape((B * C, Z, Y, X))
        for i in range(flat.shape[0]):
            sample = flat[i]
            center = np.median(sample)
            mad = np.median(np.abs(sample - center))
            scale = (mad * 1.4826) if mad > 1e-6 else 1.0
            sample_norm = (sample - center) / scale
            flat[i] = np.clip(sample_norm, -4.0, 4.0)
        out = flat.reshape(B, C, Z, Y, X)

    elif method == 'zscore':
        mean = data.mean(axis=(2, 3, 4), keepdims=True)
        std = data.std(axis=(2, 3, 4), keepdims=True)
        std[std < eps] = 1.0
        out = (data - mean) / std

    else:
        raise ValueError(f"不支持的归一化方式: {method}")

    # 还原到输入形状
    if patch.ndim == 5:
        return out.astype(np.float32)
    elif patch.ndim == 4:
        return out[:, 0].astype(np.float32)
    else:
        return out[0, 0].astype(np.float32)

def compute_starts(dim, psize, stride):
    """计算滑动窗口的起始位置"""
    if dim <= psize:
        return [0]
    
    starts = list(range(0, dim - psize + 1, stride))
    
    # 确保覆盖整个维度
    if starts[-1] + psize < dim:
        starts.append(dim - psize)
    
    # 去重（当stride=psize时可能有重复）
    return sorted(set(starts))

def calculate_patch_statistics(volume_shape, patch_size, stride):
    """计算patch统计信息"""
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
    avg_overlap = (overlap_z + overlap_y + overlap_x) / 3
    
    # 计算覆盖统计
    coverage_map = np.zeros(volume_shape, dtype=np.float32)
    for iz in starts_z:
        for iy in starts_y:
            for ix in starts_x:
                coverage_map[iz:iz+pz, iy:iy+py, ix:ix+px] += 1.0
    
    min_coverage = coverage_map.min()
    max_coverage = coverage_map.max()
    avg_coverage = coverage_map.mean()
    
    return {
        'total_patches': total_patches,
        'overlap_rate_z': overlap_z,
        'overlap_rate_y': overlap_y,
        'overlap_rate_x': overlap_x,
        'avg_overlap': avg_overlap,
        'min_coverage': min_coverage,
        'max_coverage': max_coverage,
        'avg_coverage': avg_coverage,
        'starts_z': starts_z,
        'starts_y': starts_y,
        'starts_x': starts_x,
        'coverage_map': coverage_map
    }

def adjust_volume_dimensions(volume, target_shape, mode='reflect'):
    """
    将体积调整到目标形状
    如果当前尺寸大，居中裁剪；如果小，填充
    """
    if volume.ndim != 3:
        raise ValueError(f"期望3D体积，得到 shape={volume.shape}")
    
    current_z, current_y, current_x = volume.shape
    target_z, target_y, target_x = target_shape
    
    adjustments = []
    
    # 检查并调整X维度
    if current_x != target_x:
        if current_x > target_x:
            # 裁剪X维度
            start_x = (current_x - target_x) // 2
            end_x = start_x + target_x
            volume = volume[:, :, start_x:end_x]
            adjustments.append(f"X裁剪: {current_x}->{target_x}")
        else:
            # 填充X维度
            pad_before = (target_x - current_x) // 2
            pad_after = target_x - current_x - pad_before
            volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode=mode)
            adjustments.append(f"X填充: {current_x}->{target_x}")
    
    # 调整Z维度
    current_z, current_y, _ = volume.shape
    if current_z > target_z:
        start_z = (current_z - target_z) // 2
        end_z = start_z + target_z
        volume_z = volume[start_z:end_z, :, :]
        adjustments.append(f"Z裁剪: {current_z}->{target_z}")
    elif current_z < target_z:
        pad_before = (target_z - current_z) // 2
        pad_after = target_z - current_z - pad_before
        volume_z = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode=mode)
        adjustments.append(f"Z填充: {current_z}->{target_z}")
    else:
        volume_z = volume
    
    # 调整Y维度
    current_z2, current_y2, _ = volume_z.shape
    if current_y2 > target_y:
        start_y = (current_y2 - target_y) // 2
        end_y = start_y + target_y
        volume_zy = volume_z[:, start_y:end_y, :]
        adjustments.append(f"Y裁剪: {current_y2}->{target_y}")
    elif current_y2 < target_y:
        pad_before = (target_y - current_y2) // 2
        pad_after = target_y - current_y2 - pad_before
        volume_zy = np.pad(volume_z, ((0, 0), (pad_before, pad_after), (0, 0)), mode=mode)
        adjustments.append(f"Y填充: {current_y2}->{target_y}")
    else:
        volume_zy = volume_z
    
    if adjustments:
        print("维度调整: " + ", ".join(adjustments))
    
    return volume_zy

def sliding_inference_high_precision(volume, model, device, patch_size, stride, 
                                    normalization='minmax', norm_range=(-1.0, 1.0),
                                    batch_infer=1, use_amp=True, save_overlap=False):
    """
    高精度滑动窗口推理（25%重叠率）
    使用优化的内存管理和批处理
    """
    Z, Y, X = volume.shape
    pz, py, px = patch_size
    stz, sty, stx = stride
    
    # 计算patch统计
    stats = calculate_patch_statistics((Z, Y, X), patch_size, stride)
    
    print(f"体积大小: {volume.shape}")
    print(f"Patch大小: {patch_size}")
    print(f"步长: {stride}")
    print(f"总patch数: {stats['total_patches']:,}")
    print(f"重叠率: Z={stats['overlap_rate_z']:.1%}, Y={stats['overlap_rate_y']:.1%}, X={stats['overlap_rate_x']:.1%}")
    print(f"覆盖范围: 最小{stats['min_coverage']:.1f}, 最大{stats['max_coverage']:.1f}, 平均{stats['avg_coverage']:.1f}")
    
    # 预计算所有坐标
    coords = [(iz, iy, ix) for iz in stats['starts_z'] 
                            for iy in stats['starts_y'] 
                            for ix in stats['starts_x']]
    
    # 初始化结果数组（使用float32保持精度）
    prob_sum = np.zeros((Z, Y, X), dtype=np.float32)
    count = np.zeros((Z, Y, X), dtype=np.float32) if save_overlap else None
    
    model.eval()
    device = torch.device(device)
    use_amp = use_amp and (device.type == 'cuda')
    
    # 性能统计
    total_time = 0
    patch_times = []
    
    # 批量预测
    with torch.no_grad():
        progress_bar = tqdm(total=len(coords), desc='推理进度', unit='patch')
        
        i = 0
        while i < len(coords):
            batch_start_time = time.time()
            
            # 动态调整batch大小
            current_batch_size = min(batch_infer, len(coords) - i)
            batch_coords = coords[i:i + current_batch_size]
            
            # 准备batch数据
            batch_patches = []
            orig_shapes = []
            
            for (iz, iy, ix) in batch_coords:
                # 提取patch（确保不越界）
                z_end = min(iz + pz, Z)
                y_end = min(iy + py, Y)
                x_end = min(ix + px, X)
                
                patch = volume[iz:z_end, iy:y_end, ix:x_end]
                orig_shapes.append(patch.shape)
                
                # 如果patch小于目标尺寸，填充
                if patch.shape != (pz, py, px):
                    pad_z = pz - patch.shape[0]
                    pad_y = py - patch.shape[1]
                    pad_x = px - patch.shape[2]
                    
                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        pad_width = ((0, pad_z), (0, pad_y), (0, pad_x))
                        patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
                
                batch_patches.append(patch[None, ...])  # 添加通道维度
            
            # 堆叠成batch并归一化
            if batch_patches:
                batch_np = np.concatenate(batch_patches, axis=0)  # (B, Z, Y, X)
                
                # 批量归一化（提高效率）
                batch_norm = normalize_patch(batch_np, method=normalization, 
                                            norm_range=norm_range)
                
                # 添加通道维度
                batch_norm = batch_norm[:, None, ...]  # (B, 1, Z, Y, X)
                
                batch_tensor = torch.from_numpy(batch_norm).float().to(device)
                
                # 推理
                try:
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits = model(batch_tensor)
                            probs = torch.sigmoid(logits)
                    else:
                        logits = model(batch_tensor)
                        probs = torch.sigmoid(logits)
                    
                    probs_np = probs.cpu().numpy()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nGPU内存不足，尝试减小batch_infer")
                        torch.cuda.empty_cache()
                        if batch_infer > 1:
                            batch_infer = max(1, batch_infer // 2)
                            print(f"自动减小batch_infer为: {batch_infer}")
                            continue
                    raise e
                
                # 处理输出形状
                if probs_np.ndim == 5 and probs_np.shape[1] == 1:
                    probs_np = probs_np[:, 0, ...]  # (B, Z, Y, X)
                
                # 累加概率到对应位置
                for bi, (iz, iy, ix) in enumerate(batch_coords):
                    prob_patch = probs_np[bi]
                    orig_shape = orig_shapes[bi]
                    
                    # 截取原始大小（去除填充部分）
                    prob_patch_crop = prob_patch[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
                    
                    # 累加
                    z_end = iz + orig_shape[0]
                    y_end = iy + orig_shape[1]
                    x_end = ix + orig_shape[2]
                    
                    prob_sum[iz:z_end, iy:y_end, ix:x_end] += prob_patch_crop
                    
                    if save_overlap:
                        count[iz:z_end, iy:y_end, ix:x_end] += 1.0
            
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
    
    # 计算平均概率
    print(f"\n推理完成，总耗时: {total_time:.2f}秒")
    print(f"平均每patch: {np.mean(patch_times):.3f}秒")
    
    if save_overlap:
        # 使用实际的count数组
        valid_mask = count > 0
        prob_avg = np.zeros_like(prob_sum, dtype=np.float32)
        prob_avg[valid_mask] = prob_sum[valid_mask] / count[valid_mask]
        
        # 检查覆盖情况
        uncovered = np.sum(~valid_mask)
        if uncovered > 0:
            print(f"警告: {uncovered}个体素未被任何patch覆盖 ({uncovered/valid_mask.size:.2%})")
        else:
            print("所有体素都被至少一个patch覆盖")
        
        return prob_avg, count, stats['coverage_map']
    else:
        # 使用预计算的覆盖图
        coverage = stats['coverage_map']
        valid_mask = coverage > 0
        prob_avg = np.zeros_like(prob_sum, dtype=np.float32)
        prob_avg[valid_mask] = prob_sum[valid_mask] / coverage[valid_mask]
        
        return prob_avg, None, stats['coverage_map']

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

def main():
    # 打印配置摘要
    print_config_summary()
    
    # 检查输入文件
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取数据
    print(f"\n1. 读取数据...")
    vol = read_volume(str(input_path), expected_shape=expected_shape, 
                     expected_order=expected_order)
    
    if vol.ndim != 3:
        raise ValueError(f"读取后不是3D数组，shape={vol.shape}")
    
    print(f"数据形状: {vol.shape}")
    print(f"数据范围: 最小值={vol.min():.6f}, 最大值={vol.max():.6f}, 均值={vol.mean():.6f}")
    
    # # 归一化
    # if normalization != 'none':
    #     print(f"\n2. 数据归一化 ({normalization})...")
    #     normalize_start = time.time()
    #     vol = normalize_patch(vol, method=normalization, norm_range=normalize_range)
    #     normalize_time = time.time() - normalize_start
    #     print(f"归一化完成，耗时: {normalize_time:.2f}秒")
    #     print(f"归一化后范围: 最小值={vol.min():.6f}, 最大值={vol.max():.6f}")
    
    # 调整维度到期望形状
    if vol.shape != expected_shape:
        print(f"\n3. 调整体积维度...")
        adjust_start = time.time()
        vol = adjust_volume_dimensions(vol, expected_shape, mode='reflect')
        adjust_time = time.time() - adjust_start
        print(f"调整完成，耗时: {adjust_time:.2f}秒")
        print(f"最终形状: {vol.shape}")
    
    # 初始化模型
    print(f"\n4. 初始化模型...")
    # 模型注册表：如需新增模型，在此加入对应的 key -> class/constructor
    MODEL_REGISTRY = {
        'unet3d': UNet3D,
        'aerb_light': AERBUNet3DLight,
        'attn_light': LightAttentionUNet3D,
        'aerb': AERBUNet3D,
        'seunet': SEUNet3D,
        'attention_unet3d': AttentionUNet3D,
    }

    ModelClass = MODEL_REGISTRY.get(model_name, UNet3D)
    # 尝试几组常见构造参数以兼容不同模型签名
    model = None
    for kwargs in ({'in_channels': 1, 'out_channels': 1}, {'in_channels': 1, 'base_channels': 16}, {}):
        try:
            model = ModelClass(**kwargs)
            break
        except Exception:
            continue
    if model is None:
        # 回退到默认 UNet3D
        model = UNet3D(in_channels=1, out_channels=1)

    device_t = torch.device(device)
    model.to(device_t)
    
    # 使用与 evaluate.py 相同的智能查找 checkpoint 策略（优先 latest/*best* -> latest/*last* -> timestamp/*best* -> 全局回退）

    ckpt_to_use = None

    def find_preferred(dir_path: Path):
        if not dir_path.exists():
            return None
        # 优先常规命名
        for name in ('model_best_iou.pth', 'model_best.pth', f'{model_name}_best.pth'):
            p = dir_path / name
            if p.exists():
                return p
        # 其次任何 best
        bests = sorted(dir_path.glob('*best*.pth'), key=lambda p: p.stat().st_mtime)
        if bests:
            return bests[-1]
        # 再次 last
        for name in ('model_last.pth', f'{model_name}_last.pth'):
            p = dir_path / name
            if p.exists():
                return p
        lasts = sorted(dir_path.glob('*last*.pth'), key=lambda p: p.stat().st_mtime)
        if lasts:
            return lasts[-1]
        return None

    cp = Path(checkpoint_path) if checkpoint_path is not None else None
    if cp is not None and cp.exists() and cp.is_file():
        ckpt_to_use = cp
    else:
        # 目录候选：<model_name> 与 <model_name>_c16
        candidate_dirs = [model_name, f'{model_name}_c16']
        for sub in candidate_dirs:
            cand_latest = checkpoints_root / sub / 'latest'
            ckpt_to_use = find_preferred(cand_latest)
            if ckpt_to_use is not None:
                break
        # 按时间戳子目录回退
        if ckpt_to_use is None:
            for sub in candidate_dirs:
                model_root = checkpoints_root / sub
                if model_root.exists():
                    subdirs = sorted([d for d in model_root.iterdir() if d.is_dir() and d.name != 'latest'],
                                     key=lambda d: d.stat().st_mtime)
                    for ts_dir in reversed(subdirs):
                        ckpt_to_use = find_preferred(ts_dir)
                        if ckpt_to_use is not None:
                            break
                if ckpt_to_use is not None:
                    break
        # 全局兜底（不推荐，但避免直接失败）
        if ckpt_to_use is None and checkpoints_root.exists():
            scoped = sorted(checkpoints_root.glob(f'**/{model_name}*/**/*best*.pth'),
                            key=lambda p: p.stat().st_mtime)
            if scoped:
                ckpt_to_use = scoped[-1]
            else:
                scoped_last = sorted(checkpoints_root.glob(f'**/{model_name}*/**/*last*.pth'),
                                     key=lambda p: p.stat().st_mtime)
                if scoped_last:
                    ckpt_to_use = scoped_last[-1]

    
    if ckpt_to_use is None or not Path(ckpt_to_use).exists():
        raise FileNotFoundError(f'未找到 {model_name} 的checkpoint；请检查 checkpoints2/{model_name}[_c16]/latest/')
    
    # 【修改点】使用自动找到的 ckpt_to_use，而不是空的 checkpoint_path
    final_ckpt = Path(ckpt_to_use)
    
    if final_ckpt.exists():
        model = load_model_weights(model, final_ckpt, device_t)
        # 【关键】把找到的路径赋值回 checkpoint_path，这样后面的代码（如保存文件名时）就不会报错了
    
    else:
        raise FileNotFoundError(f'Checkpoint未找到: {final_ckpt}')
    
    # 推理
    print(f"\n5. 开始高精度滑动窗口推理...")
    print(f"注意: 使用25%重叠率，patch数量较多，请耐心等待")
    
    inference_start = time.time()
    result = sliding_inference_high_precision(
        volume=vol,
        model=model,
        device=device_t,
        patch_size=patch_size,
        stride=stride,
        normalization=normalization,
        norm_range=normalize_range,
        batch_infer=batch_infer,
        use_amp=use_amp,
        save_overlap=save_overlap_map
    )
    inference_time = time.time() - inference_start
    
    if save_overlap_map:
        prob_map, overlap_map, coverage_map = result
    else:
        prob_map, _, coverage_map = result
    
    print(f"\n推理完成! 总耗时: {inference_time/60:.1f}分钟")
    
    # 分析结果
    print(f"\n6. 结果分析...")
    print(f"概率图范围: 最小值={prob_map.min():.6f}, 最大值={prob_map.max():.6f}")
    print(f"概率图统计: 均值={prob_map.mean():.6f}, 标准差={prob_map.std():.6f}")
    print(f"覆盖统计: 最小重叠={coverage_map.min()}, 最大重叠={coverage_map.max()}, 平均重叠={coverage_map.mean():.2f}")
    
    # 二值化
    mask = (prob_map >= threshold).astype(np.uint8)
    pos_ratio = mask.mean()
    print(f"二值化统计: 阈值={threshold}, 正样本比例={pos_ratio:.4%}")
    print(f"             正样本数={mask.sum():,}, 负样本数={mask.size - mask.sum():,}")
    
    # 保存结果
    print(f"\n7. 保存结果...")

    # 使用 checkpoint 名称作为模型名（如 unet3d_best），并用时间戳区分每次运行
    save_model_name = final_ckpt.stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 新增：以输入文件名为第一层目录
    input_name = Path(input_path).stem
    output_dir = Path('outputs1') / input_name / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 二值mask（保留）
    mask_path = output_dir / out_mask_npy.name
    np.save(mask_path, mask)
    print(f"✓ 二值mask保存到: {mask_path}")

    # 保存处理摘要（json）（保留）
    summary = {
        'model': save_model_name,
        'timestamp': timestamp,
        'input': str(input_path),
        'expected_shape': expected_shape,
        'patch_size': patch_size,
        'stride': stride,
        'threshold': float(threshold),
        'total_time_s': float(inference_time),
        'mask_path': str(mask_path),
    }
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ 处理摘要保存到: {summary_path}")
    print(f"\n8. 生成 SEG-Y 文件 (使用固定原始文件)...")
    
    # 1. 配置路径 (使用您指定的固定路径)
    # 这里的 mask_path 是上面代码刚刚生成的，直接拿来用
    ORIGINAL_FILE = r'2020Z205_3D_PSTM_TIME_mini_400_2600ms.sgy'
    PREDICTED_NPY = str(mask_path) 
    OUTPUT_FILE = str(output_dir / 'predicted_result_segyio_final1.sgy')
    
    # 2. 检查文件是否存在
    if not os.path.exists(ORIGINAL_FILE):
        print(f"  错误：原始 SEG-Y 文件未找到：{ORIGINAL_FILE}")
        print("  -> 跳过 SEG-Y 生成。")
    elif not os.path.exists(PREDICTED_NPY):
        print(f"  错误：预测 NumPy 文件未找到：{PREDICTED_NPY}")
    else:
        try:
            print(f"  原始 SGY: {ORIGINAL_FILE}")
            print(f"  预测 NPY: {PREDICTED_NPY}")
            
            # --- 步骤 1: 准备数据 ---
            # 加载预测数据
            predicted_data_np = np.load(PREDICTED_NPY).astype(np.float32)
            
            # 根据您的代码逻辑：shape[2] 是采样点数 (Samples)
            # 如果生成的数据形状不对，这里可能会报错，需要留意
            N_samples = predicted_data_np.shape[2]
            predicted_data_2d = predicted_data_np.reshape(-1, N_samples)
            N_traces, _ = predicted_data_2d.shape
            
            print(f"  -> 数据已重塑: ({N_traces} 道, {N_samples} 采样点)")
            
            # --- 步骤 2: 读取原始结构 ---
            with segyio.open(ORIGINAL_FILE, ignore_geometry=True) as src:
                spec = segyio.spec()
                spec.ilines = src.ilines
                spec.xlines = src.xlines
                spec.samples = src.samples
                spec.format = 5     # 4-byte IEEE float
                spec.sorting = 0    # NotSorted
                spec.tracecount = N_traces
                
                original_tracecount = src.tracecount
                if original_tracecount != N_traces:
                    print(f"  警告：原始文件道数 ({original_tracecount}) 与预测 ({N_traces}) 不匹配")
                
                N_copy_traces = min(original_tracecount, N_traces)
                
                # --- 步骤 3: 写入新文件 ---
                print(f"  正在写入: {OUTPUT_FILE} ...")
                with segyio.create(OUTPUT_FILE, spec) as dst:
                    # 复制头信息
                    dst.text[0] = src.text[0]
                    dst.bin = src.bin
                    
                    # 写入数据和道头 (使用 tqdm 显示进度)
                    for i in tqdm(range(N_traces), desc='  写入SEG-Y', unit='trace'):
                        dst.trace[i] = predicted_data_2d[i]
                        
                        # 复制/构造道头
                        if i < N_copy_traces:
                            dst.header[i] = src.header[i]
                        else:
                            if original_tracecount > 0:
                                dst.header[i] = src.header[0]
                                dst.header[i][segyio.cdp] = i + 1
                                dst.header[i][segyio.traceno] = i + 1
                            else:
                                dst.header[i][segyio.cdp] = i + 1
                                dst.header[i][segyio.traceno] = i + 1
                                
            print(f"✓ SEG-Y 文件生成成功: {OUTPUT_FILE}")
            
        except Exception as e:
            print("-" * 30)
            print(f"使用 segyio 转换时发生错误: {e}")
            print(f"当前数据形状: {predicted_data_np.shape}")
            import traceback
            traceback.print_exc()

    print(f"\n所有输出已保存到目录: {output_dir}")
    
    print("\n" + "=" * 70)
    print("高精度推理完成!")
    print("=" * 70)
    print(f"关键信息:")
    print(f"  • 总patch数: {len(compute_starts(vol.shape[0], patch_size[0], stride[0])) * len(compute_starts(vol.shape[1], patch_size[1], stride[1])) * len(compute_starts(vol.shape[2], patch_size[2], stride[2])):,}")
    print(f"  • 推理时间: {inference_time/60:.1f} 分钟")
    print(f"  • 正样本比例: {pos_ratio:.4%}")
    print(f"  • 输出文件:")
    print(f"      - {out_prob_npy} (概率图)")
    print(f"      - {out_mask_npy} (二值分割)")
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