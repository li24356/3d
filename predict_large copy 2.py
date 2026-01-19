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
# 新增依赖：用于滞后阈值
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize
# ---------- 模型导入 ----------
from models.unet3d import UNet3D
from models.AERB3d import AERBUNet3D
from models.AERB3d import AERBUNet3DLight
from models.attention_unet3d import LightAttentionUNet3D
from models.seunet3d import SEUNet3D
from models.attention_unet3d import AttentionUNet3D
from models.AERB_pro import AERBPRO

# ---------- 核心配置 ----------
input_path = Path(r'F3data.npy')
checkpoint_path = None 
checkpoints_root = Path('checkpoints3')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 维度配置
expected_shape = (601,951,391)
expected_order = 'C'

model_name = 'aerb_pro'   # 可改为 'attn_light' / 'aerb_light' / 你的自定义 key/ 'unet3d'

# 模型配置
patch_size = (128, 128, 128)

# [优化1] 滑动窗口：50% 重叠
stride = (64, 64, 64)
overlap_rate = 0.5

# [优化2] TTA 配置
enable_tta = True             # 【开启TTA】：会进行 [原图, 翻转X, 翻转Y] 3次预测取平均
batch_infer = 1               # 开启TTA时建议减小Batch，避免显存溢出

# [优化3] 后处理配置
use_hysteresis = True         # 【开启滞后阈值】
threshold_high = 0.6          # 强断层阈值 (确定是断层)
threshold_low = 0.3           # 弱断层阈值 (如果是连接着强断层的，也算)

# 其他配置
use_amp = True
out_prob_npy = Path('pred_prob.npy')
out_mask_npy = Path('pred_mask.npy')
save_overlap_map = False      # 关闭这个可以省点内存

# -------------------------------------------------------
#   辅助函数
# -------------------------------------------------------

def get_gaussian_weight_map(patch_size, sigma_scale=1.0/8):
    """生成3D高斯权重图"""
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[center_coords[0], center_coords[1], center_coords[2]] = 1
    weight_map = scipy.ndimage.gaussian_filter(tmp, sigmas)
    weight_map = weight_map / weight_map.max()
    return torch.from_numpy(weight_map).float()

def _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0), device=None):
    """GPU版鲁棒归一化"""
    if not isinstance(x, torch.Tensor): x = torch.from_numpy(np.asarray(x)).float()
    if device is None: device = x.device
    x = x.to(device)
    
    if x.ndim == 5: n_samples = x.shape[0] * x.shape[1]
    else: n_samples = x.shape[0]
    
    x_reshaped = x.reshape(n_samples, -1)
    center = torch.median(x_reshaped, dim=1, keepdim=True)[0]
    
    if use_mad:
        abs_dev = torch.abs(x_reshaped - center)
        mad = torch.median(abs_dev, dim=1, keepdim=True)[0]
        scale = torch.where(mad > 1e-6, mad * 1.4826, torch.ones_like(mad))
    else:
        scale = torch.std(x_reshaped, dim=1, keepdim=True)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    
    x_norm = (x_reshaped - center) / scale
    if clip_range: x_norm = torch.clamp(x_norm, clip_range[0], clip_range[1])
    return x_norm.reshape(x.shape).float()

def read_volume(path, expected_shape=None, expected_order='C'):
    path = Path(path)
    if path.suffix == '.npy':
        arr = np.load(path)
    elif path.suffix == '.npz':
        with np.load(path) as data: arr = data[data.files[0]]
    else: raise ValueError("Unknown format")
    
    if arr.ndim == 2 and expected_shape:
        arr = arr.reshape(expected_shape, order=expected_order)
    return arr.astype(np.float32)

def adjust_volume_dimensions(volume, target_shape):
    # 简化版维度调整
    pads = []
    slices = []
    for c, t in zip(volume.shape, target_shape):
        if c < t:
            pads.append(((t-c)//2, t-c-(t-c)//2))
            slices.append(slice(None))
        else:
            pads.append((0,0))
            start = (c-t)//2
            slices.append(slice(start, start+t))
    
    if any(p != (0,0) for p in pads): volume = np.pad(volume, pads, 'reflect')
    if any(s != slice(None) for s in slices): volume = volume[tuple(slices)]
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
#   核心推理逻辑 (含 TTA)
# -------------------------------------------------------

def sliding_inference_full(volume, model, device, patch_size, stride, 
                           batch_infer=1, use_amp=True, enable_tta=False):
    
    Z, Y, X = volume.shape
    pz, py, px = patch_size
    
    # 1. 准备高斯权重
    weight_map = get_gaussian_weight_map(patch_size, sigma_scale=1.0/8).to(device)
    
    # 计算切片位置
    starts_z = list(range(0, Z - pz + 1, stride[0]))
    if starts_z[-1] + pz < Z: starts_z.append(Z - pz)
    
    starts_y = list(range(0, Y - py + 1, stride[1]))
    if starts_y[-1] + py < Y: starts_y.append(Y - py)
    
    starts_x = list(range(0, X - px + 1, stride[2]))
    if starts_x[-1] + px < X: starts_x.append(X - px)
    
    coords = [(iz, iy, ix) for iz in starts_z for iy in starts_y for ix in starts_x]
    
    print(f"推理配置: Patch={patch_size}, Stride={stride}, TTA={'开启' if enable_tta else '关闭'}")
    print(f"总Patch数: {len(coords)}")

    # 2. 初始化结果容器 (尝试GPU)
    try:
        prob_sum = torch.zeros((Z, Y, X), device=device, dtype=torch.float32)
        count_sum = torch.zeros((Z, Y, X), device=device, dtype=torch.float32)
        on_gpu_accum = True
    except RuntimeError:
        print("显存不足，使用CPU累加...")
        prob_sum = torch.zeros((Z, Y, X), dtype=torch.float32)
        count_sum = torch.zeros((Z, Y, X), dtype=torch.float32)
        on_gpu_accum = False

    model.eval()
    
    with torch.no_grad():
        progress_bar = tqdm(total=len(coords), desc='Inference', unit='patch')
        i = 0
        while i < len(coords):
            batch_coords = coords[i:i + batch_infer]
            
            # --- 准备数据 ---
            batch_tensors = []
            orig_shapes = []
            
            for (iz, iy, ix) in batch_coords:
                z_end, y_end, x_end = min(iz + pz, Z), min(iy + py, Y), min(ix + px, X)
                patch = volume[iz:z_end, iy:y_end, ix:x_end]
                orig_shapes.append(patch.shape)
                
                # Padding
                if patch.shape != (pz, py, px):
                    patch = np.pad(patch, ((0, pz-patch.shape[0]), (0, py-patch.shape[1]), (0, px-patch.shape[2])), 'constant')
                
                batch_tensors.append(torch.from_numpy(patch))
            
            # (B, 1, Z, Y, X)
            batch_input = torch.stack(batch_tensors).float().unsqueeze(1).to(device)
            
            # GPU 归一化
            batch_input = _robust_normalize_tensor_batch(batch_input, device=device)
            
            # --- TTA 推理逻辑 ---
            # 基础预测
            inputs_list = [batch_input]
            flip_dims = []
            
            if enable_tta:
                # 添加翻转版本: dim=3 (Y轴), dim=4 (X轴)
                # 注意 input 是 (B, C, Z, Y, X)，所以 Y=3, X=4
                inputs_list.append(torch.flip(batch_input, [3])) # Flip Y
                inputs_list.append(torch.flip(batch_input, [4])) # Flip X
                flip_dims = [None, [3], [4]]
            else:
                flip_dims = [None]
            
            # 依次预测并融合
            avg_probs = None
            
            for idx, inp in enumerate(inputs_list):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = torch.sigmoid(model(inp))
                else:
                    out = torch.sigmoid(model(inp))
                
                # 如果是翻转过的，需要翻转回来
                if flip_dims[idx] is not None:
                    out = torch.flip(out, flip_dims[idx])
                
                if avg_probs is None:
                    avg_probs = out
                else:
                    avg_probs += out
            
            # 取平均
            probs = avg_probs / len(inputs_list)
            
            # --- 加权融合 ---
            probs = probs.squeeze(1) # (B, Z, Y, X)
            weighted_probs = probs * weight_map
            
            if not on_gpu_accum:
                weighted_probs = weighted_probs.cpu()
                w_map_cpu = weight_map.cpu()

            for bi, (iz, iy, ix) in enumerate(batch_coords):
                os = orig_shapes[bi]
                z_end, y_end, x_end = iz + os[0], iy + os[1], ix + os[2]
                
                valid_prob = weighted_probs[bi, :os[0], :os[1], :os[2]]
                
                if on_gpu_accum:
                    valid_weight = weight_map[:os[0], :os[1], :os[2]]
                    prob_sum[iz:z_end, iy:y_end, ix:x_end] += valid_prob
                    count_sum[iz:z_end, iy:y_end, ix:x_end] += valid_weight
                else:
                    valid_weight = w_map_cpu[:os[0], :os[1], :os[2]]
                    prob_sum[iz:z_end, iy:y_end, ix:x_end] += valid_prob
                    count_sum[iz:z_end, iy:y_end, ix:x_end] += valid_weight

            i += len(batch_coords)
            progress_bar.update(len(batch_coords))
            
        progress_bar.close()
    
    count_sum = torch.clamp(count_sum, min=1e-6)
    final_prob = prob_sum / count_sum
    
    return final_prob.cpu().numpy()

# -------------------------------------------------------
#   主程序
# -------------------------------------------------------

def main():
    print("="*50)
    print("3D Seismic Inference - Ultimate Version")
    print(f"Features: Gaussian Blending | TTA={enable_tta} | Hysteresis={use_hysteresis}")
    print("="*50)
    
    # 1. 读取
    if not input_path.exists(): raise FileNotFoundError("Input not found")
    vol = read_volume(input_path, expected_shape, expected_order)
    if vol.shape != expected_shape: vol = adjust_volume_dimensions(vol, expected_shape)
    
    # 2. 模型
    MODEL_REGISTRY = {
        'unet3d': UNet3D, 'aerb_light': AERBUNet3DLight, 'attn_light': LightAttentionUNet3D,
        'aerb': AERBUNet3D, 'seunet': SEUNet3D, 'attention_unet3d': AttentionUNet3D,
        'aerb_pro': AERBPRO,
    }
    ModelClass = MODEL_REGISTRY.get(model_name, UNet3D)
    
    # 特殊处理：AERB_pro 使用 base_channels=32（与训练一致）
    if model_name == 'aerb_pro':
        try: 
            model = ModelClass(in_channels=1, out_channels=1, base_channels=32)
        except: 
            model = ModelClass(in_channels=1, base_channels=32)
    else:
        try: 
            model = ModelClass(in_channels=1, out_channels=1)
        except: 
            model = ModelClass(in_channels=1, base_channels=16)
    
    device_t = torch.device(device)
    model.to(device_t)
    
    # 3. Checkpoint (简化查找逻辑)
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
    else:
        # 自动查找逻辑
        search_dir = checkpoints_root / model_name / 'latest'
        ckpt = list(search_dir.glob('*best*.pth'))
        if not ckpt: ckpt = list(checkpoints_root.glob(f'**/{model_name}*/**/*best*.pth'))
        if not ckpt: raise FileNotFoundError("No checkpoint found")
        ckpt = ckpt[-1] # 取最新的
    
    print(f"Loading checkpoint: {ckpt}")
    model = load_model_weights(model, ckpt, device_t)
    
    # 4. 推理
    start_t = time.time()
    prob_map = sliding_inference_full(
        vol, model, device_t, patch_size, stride, 
        batch_infer=batch_infer, use_amp=use_amp, enable_tta=enable_tta
    )
    print(f"Inference Time: {(time.time()-start_t)/60:.1f} min")
    
    # 5. 后处理 (含滞后阈值)
    print("Post-processing...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 输出路径: outputs / input_name / model_name / timestamp
    save_dir = Path('outputs') / input_path.stem / model_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 策略 A: 调整阈值 (解决粘连的第一步) ---
    # 既然出现了“大黑团”，说明模型预测太自信了，或者对背景的抑制不够
    # 建议提高低阈值，比如从 0.3 提到 0.45 或 0.5
    final_low_thresh = 0.45 
    final_high_thresh = 0.70
    
    print(f"1. 应用滞后阈值 (Low={final_low_thresh}, High={final_high_thresh})...")
    # 先生成比较“粗”的二值图
    mask_thick = apply_hysteresis_threshold(prob_map, final_low_thresh, final_high_thresh)
    
    # --- 策略 B: 骨架化 (解决粘连的终极手段) ---
    print("2. 执行3D骨架化 (Skeletonization) - 瘦身处理...")
    # skeletonize 输入必须是 bool 类型，且计算量较大，需要一点时间
    # 它会将粗大的断层剥蚀成单像素宽的线/面
    mask_thin = skeletonize(mask_thick)
    
    # 转回 uint8 用于保存
    mask_thick_uint8 = mask_thick.astype(np.uint8)
    mask_thin_uint8 = mask_thin.astype(np.uint8)

    # 统计正样本比例并打印
    pos_ratio = float(mask_thin_uint8.mean())
    pos_count = int(mask_thin_uint8.sum())
    neg_count = int(mask_thin_uint8.size - pos_count)
    print(f"正样本比例: {pos_ratio:.4%} (正样本数={pos_count:,}, 负样本数={neg_count:,})")
    
    # 保存文件
    print("保存结果...")
    np.save(save_dir / out_prob_npy, prob_map)
    np.save(save_dir / 'mask_thick.npy', mask_thick_uint8) # 保存一份未瘦身的以备对比
    np.save(save_dir / out_mask_npy, mask_thin_uint8)      # 最终结果是瘦身后的

    # 保存处理摘要（json）
    summary = {
        'model': model_name,
        'checkpoint': str(ckpt),
        'timestamp': timestamp,
        'input': str(input_path),
        'expected_shape': expected_shape,
        'patch_size': patch_size,
        'stride': stride,
        'threshold_low': float(final_low_thresh),
        'threshold_high': float(final_high_thresh),
        'pos_ratio': pos_ratio,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'prob_path': str(save_dir / out_prob_npy),
        'mask_path': str(save_dir / out_mask_npy),
        'mask_thick_path': str(save_dir / 'mask_thick.npy'),
    }
    with open(save_dir / 'inference_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 6. SEGY 导出
    print("Generating SEGY...")
    ORIG_SGY = os.path.splitext(str(input_path))[0] + '.sgy'
    if os.path.exists(ORIG_SGY):
        try:
            # 快速重塑
            pred_2d = mask_thin.reshape(-1, mask_thin.shape[2])
            spec = segyio.spec()
            # 读取原始头信息
            with segyio.open(ORIG_SGY, ignore_geometry=True) as src:
                spec.ilines, spec.xlines = src.ilines, src.xlines
                spec.samples = src.samples
                spec.format = 5
                spec.tracecount = pred_2d.shape[0]
                
                dst_path = save_dir / f"{input_path.stem}_pred.sgy"
                with segyio.create(str(dst_path), spec) as dst:
                    dst.text[0] = src.text[0]
                    # 批量写入可能需要大内存，这里逐道写入更稳妥
                    for i in tqdm(range(len(pred_2d)), desc="Writing SEGY"):
                        dst.trace[i] = pred_2d[i].astype(np.float32)
                        # 简单的头信息复制
                        if i < src.tracecount: dst.header[i] = src.header[i]
                        else: 
                            dst.header[i] = src.header[0]
                            dst.header[i][segyio.cdp] = i+1
            print(f"SEGY saved to {dst_path}")
        except Exception as e:
            print(f"SEGY Error: {e}")
    
    print(f"Done! All results in {save_dir}")

if __name__ == "__main__":
    main()