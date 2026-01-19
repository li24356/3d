import os
from pathlib import Path
from models.AERB3d import AERBUNet3D
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from models.unet3d import UNet3D
from dataloader import VolumeDataset
from models.AERB3d import AERBUNet3DLight
from models.attention_unet3d import LightAttentionUNet3D
from models.attention_unet3d import AttentionUNet3D
from models.seunet3d import SEUNet3D
from models.AERB_pro import AERBPRO
# ============================================================================
# 归一化函数（必须与训练一致）
# ============================================================================
def _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0)):
    """
    鲁棒Z-score标准化：使用中位数和MAD（中位数绝对偏差）
    与训练代码保持一致
    """
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.asarray(x)
    
    original_shape = x_np.shape
    original_ndim = x_np.ndim
    
    # 重塑以便于计算
    if original_ndim == 5:
        # (B, C, Z, Y, X) -> (B*C, Z, Y, X)
        spatial_axes = (2, 3, 4)
        batch_channel_axis = (0, 1)
        n_samples = original_shape[0] * original_shape[1]
        reshaped_shape = (n_samples,) + original_shape[2:]
        x_reshaped = x_np.reshape(reshaped_shape)
    elif original_ndim == 4:
        # (B, Z, Y, X) -> (B, Z, Y, X)
        spatial_axes = (1, 2, 3)
        batch_axis = (0,)
        n_samples = original_shape[0]
        x_reshaped = x_np
    else:
        raise ValueError('Unsupported input ndim for normalization: %d' % original_ndim)
    
    # 初始化输出数组
    x_norm = np.zeros_like(x_reshaped, dtype=np.float32)
    
    # 对每个样本独立进行鲁棒归一化
    for i in range(n_samples):
        sample = x_reshaped[i]
        
        # 使用中位数作为中心（对异常值鲁棒）
        center = np.median(sample)
        
        if use_mad:
            # 使用MAD（中位数绝对偏差）作为尺度估计
            # MAD = median(|x - median(x)|)
            mad = np.median(np.abs(sample - center))
            # 将MAD转换为标准差估计: σ ≈ 1.4826 * MAD（对于正态分布）
            scale = mad * 1.4826 if mad > 1e-6 else 1.0
        else:
            # 使用标准差（传统方法）
            scale = np.std(sample)
            if scale < 1e-6:
                scale = 1.0
        
        # 鲁棒Z-score标准化
        sample_norm = (sample - center) / scale
        
        # 截断异常值
        if clip_range:
            sample_norm = np.clip(sample_norm, clip_range[0], clip_range[1])
        
        x_norm[i] = sample_norm
    
    # 恢复原始形状
    if original_ndim == 5:
        x_norm = x_norm.reshape(original_shape)
    
    return torch.from_numpy(x_norm).float()


def _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0)):
    """
    传统Z-score标准化（备选，如果训练时使用的话）
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 5:
        spatial_axes = (2, 3, 4)
    elif x.ndim == 4:
        spatial_axes = (1, 2, 3)
    else:
        raise ValueError('Unsupported input ndim for normalization: %d' % x.ndim)

    mean = x.mean(axis=spatial_axes, keepdims=True)
    std = x.std(axis=spatial_axes, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std
    
    if clip_range:
        x_norm = np.clip(x_norm, clip_range[0], clip_range[1])
    
    return torch.from_numpy(x_norm).float()


# ============================================================================
# 指标计算函数
# ============================================================================
def dice_coef(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return (2 * inter + eps) / (denom + eps)


def iou_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def precision_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)


def f1_score(pred, target, eps=1e-6):
    precision = precision_score(pred, target, eps)
    recall = recall_score(pred, target, eps)
    return (2 * precision * recall + eps) / (precision + recall + eps)


# ============================================================================
# 评估函数
# ============================================================================
def evaluate(model, loader, device, threshold=0.5, save_dir=None, use_robust_norm=True):
    """
    评估函数
    - 返回:
      summary: 各样本指标的统计(均值/方差/分位)
      stats:   各样本逐条指标列表
      micro:   微平均（聚合 TP/FP/FN/TN 后计算的一次性指标）
    """
    model.eval()
    stats = {
        'dice': [], 'iou': [], 'acc': [],
        'precision': [], 'recall': [], 'f1': []
    }

    # 微平均累计器（像素/体素级聚合）
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc='eval', leave=False)):
            # 归一化一致
            if use_robust_norm:
                x_norm = _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0)).to(device)
            else:
                x_norm = _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0)).to(device)

            y_np = np.asarray(y)
            logits = model(x_norm)
            probs = torch.sigmoid(logits).cpu().numpy()

            # (1,1,Z,Y,X) -> (Z,Y,X)
            probs = probs.squeeze(0)
            if probs.ndim == 4 and probs.shape[0] == 1:
                probs = probs[0]

            pred_bin = (probs >= threshold).astype(np.uint8)
            target_bin = (y_np.squeeze(0) > 0.5).astype(np.uint8)

            # 单样本指标（用于“均值/宏平均”）
            d = dice_coef(pred_bin, target_bin)
            j = iou_score(pred_bin, target_bin)
            acc = (pred_bin == target_bin).mean()
            p = precision_score(pred_bin, target_bin)
            r = recall_score(pred_bin, target_bin)
            f1 = f1_score(pred_bin, target_bin)

            stats['dice'].append(d)
            stats['iou'].append(j)
            stats['acc'].append(float(acc))
            stats['precision'].append(p)
            stats['recall'].append(r)
            stats['f1'].append(f1)

            # 累计 TP/FP/FN/TN（用于“微平均”）
            tp = np.logical_and(pred_bin == 1, target_bin == 1).sum()
            fp = np.logical_and(pred_bin == 1, target_bin == 0).sum()
            fn = np.logical_and(pred_bin == 0, target_bin == 1).sum()
            tn = np.logical_and(pred_bin == 0, target_bin == 0).sum()
            total_tp += int(tp)
            total_fp += int(fp)
            total_fn += int(fn)
            total_tn += int(tn)

            if save_dir:
                np.save(os.path.join(save_dir, f'prob_{idx:04d}.npy'), probs.astype(np.float32))
                np.save(os.path.join(save_dir, f'pred_{idx:04d}.npy'), pred_bin.astype(np.uint8))

    # 汇总统计（均值/宏平均）
    summary = {}
    for k, v in stats.items():
        if len(v) > 0:
            summary[k] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'min': float(np.min(v)),
                'max': float(np.max(v)),
                'median': float(np.median(v))
            }
        else:
            summary[k] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}

    # 微平均（像素/体素级聚合后一次性计算）
    micro = {}
    denom_prec = (total_tp + total_fp)
    denom_rec  = (total_tp + total_fn)
    denom_iou  = (total_tp + total_fp + total_fn)
    total_all  = (total_tp + total_fp + total_fn + total_tn)

    micro['precision'] = float(total_tp / denom_prec) if denom_prec > 0 else 0.0
    micro['recall']    = float(total_tp / denom_rec)  if denom_rec  > 0 else 0.0
    micro['f1']        = float((2*micro['precision']*micro['recall']) / (micro['precision']+micro['recall'])) if (micro['precision']+micro['recall']) > 0 else 0.0
    micro['iou']       = float(total_tp / denom_iou)  if denom_iou  > 0 else 0.0
    micro['dice']      = float((2*total_tp) / (2*total_tp + total_fp + total_fn)) if (2*total_tp + total_fp + total_fn) > 0 else 0.0
    micro['acc']       = float((total_tp + total_tn) / total_all) if total_all > 0 else 0.0

    return summary, stats, micro


# ============================================================================
# 模型加载工具
# ============================================================================
def smart_load_checkpoint_dict(ck_path, device):
    """智能加载checkpoint，处理各种格式"""
    ck = torch.load(str(ck_path), map_location=device)
    
    # 尝试多种可能的键
    possible_keys = ['model_state', 'state_dict', 'model']
    
    if isinstance(ck, dict):
        st = None
        for key in possible_keys:
            if key in ck:
                st = ck[key]
                break
        
        if st is None:
            # 如果没有找到标准键，尝试直接使用整个字典
            st = ck
    else:
        st = ck
    
    # 如果存在另一层封装，尝试展开
    if isinstance(st, dict) and 'model_state' in st:
        st = st['model_state']
    
    # 去掉 DataParallel/DistributedDataParallel 的前缀
    new_state = {}
    for k, v in st.items():
        # 移除各种前缀
        prefixes = ['module.', 'model.', '_orig_mod.']
        nk = k
        for prefix in prefixes:
            if k.startswith(prefix):
                nk = k[len(prefix):]
                break
        new_state[nk] = v
    
    return new_state


def detect_normalization_method_from_checkpoint(checkpoint_path):
    """从checkpoint中检测使用的归一化方法"""
    try:
        ck = torch.load(str(checkpoint_path), map_location='cpu')
        if isinstance(ck, dict) and 'normalization_method' in ck:
            method = ck['normalization_method']
            print(f"检测到归一化方法: {method}")
            return method
        elif isinstance(ck, dict) and 'model_config' in ck:
            print("从model_config推断归一化方法...")
            # 根据你的训练代码，如果使用鲁棒归一化会保存相关信息
            return 'robust'  # 默认假设使用鲁棒归一化
    except Exception as e:
        print(f"无法检测归一化方法: {e}")
    
    return 'robust'  # 默认使用鲁棒归一化（与训练代码一致）


# ============================================================================
# 可视化函数
# ============================================================================
def visualize_predictions(pred_files, ds_val, vis_dir, max_vis=15):
    """可视化预测结果"""
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for i, pfile in enumerate(pred_files[:max_vis]):
        try:
            # 加载预测
            pred = np.load(pfile)
            
            # 获取对应的真值
            try:
                _, y = ds_val[i]
                y = np.asarray(y).squeeze()
            except Exception:
                y = None
            
            # 处理维度
            pred3 = np.asarray(pred)
            if pred3.ndim == 4 and pred3.shape[0] == 1:
                pred3 = pred3[0]
            if y is not None and y.ndim == 4 and y.shape[0] == 1:
                y = y[0]
            
            # 计算中间切片
            shape = pred3.shape
            zc = shape[-3] // 2
            yc = shape[-2] // 2
            xc = shape[-1] // 2
            
            # 创建可视化
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            slices = [
                (pred3[zc, ...], y[zc, ...] if y is not None else None, f'Z={zc}'),
                (pred3[:, yc, :], y[:, yc, :] if y is not None else None, f'Y={yc}'),
                (pred3[:, :, xc], y[:, :, xc] if y is not None else None, f'X={xc}'),
            ]
            
            for row, (ps, ys, title) in enumerate(slices):
                # 预测图
                ax = axes[row, 0]
                ax.imshow(ps, cmap='gray')
                ax.set_title(f'Pred {title}', fontsize=10)
                ax.axis('off')
                
                # 真值图
                ax = axes[row, 1]
                if ys is not None:
                    ax.imshow(ys, cmap='gray')
                    ax.set_title('Ground Truth', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No GT', ha='center', va='center')
                ax.axis('off')
                
                # 叠加图
                ax = axes[row, 2]
                base = ys if ys is not None else ps
                ax.imshow(base, cmap='gray')
                # 添加预测轮廓
                cont = (ps >= 0.5).astype(np.uint8)
                if cont.sum() > 0:
                    try:
                        ax.contour(cont, levels=[0.5], colors=['red'], linewidths=1.0, alpha=0.7)
                    except Exception:
                        pass
                ax.set_title('Overlay (Red=Pred)', fontsize=10)
                ax.axis('off')
            
            fig.suptitle(f'Sample {i:04d}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            out_png = vis_dir / f'sample_{i:04d}_slices.png'
            fig.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"可视化样本 {i} 失败: {e}")
    
    print(f'可视化切片已保存到: {vis_dir}')




# ============================================================================
# 主函数
# ============================================================================
def main():
    # ========== 配置 ==========
    root = Path('.')
    


    # 模型选择（必须与训练时一致）
    model_name = 'aerb_pro'  # 修改为你训练时使用的模型名称
    explicit_ckpt = Path(r'checkpoints3\aerb_pro\latest\model_best_iou.pth')  # 可以显式指定路径，如 'checkpoints1/unet3d/latest/model_best_loss.pth' 
   
   
   
   
   
   
    checkpoints_root = Path('checkpoints3')




    # 归一化方法（必须与训练时一致）
    # 如果训练时使用了鲁棒归一化，这里设为True
    use_robust_norm = True  # 重要！必须与训练一致
    
    # 自动查找checkpoint

    
    # 数据配置
    dat_dtype = 'float32'
    dat_shape = (128, 128, 128)
    dat_order = 'C'
    batch_size = 1
    workers = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 评估配置
    threshold = 0.5
    report_mode = 'micro'  # 'micro' 或 'mean'，只输出一个值
    save_predictions = True
    save_dir = 'eval_predictions'
    
    # ========== 数据加载 ==========
    print('=' * 60)
    print('评估配置')
    print('=' * 60)
    print(f'模型名称: {model_name}')
    print(f'归一化方法: {"鲁棒Z-score(MAD)" if use_robust_norm else "传统Z-score"}')
    print(f'设备: {device}')
    print(f'阈值: {threshold}')
    print('=' * 60)
    
    val_data = root / 'prediction' / 'seis'
    val_label = root / 'prediction' / 'fault'
    
    if not val_data.exists() or not val_label.exists():
        print(f"错误: 验证数据路径不存在!")
        print(f"数据路径: {val_data}")
        print(f"标签路径: {val_label}")
        return
    
    ds_val = VolumeDataset(str(val_data), str(val_label), 
                          dat_dtype=dat_dtype, dat_shape=dat_shape, dat_order=dat_order)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    print(f'验证集大小: {len(ds_val)}')
    
    # ========== 模型初始化 ==========
    device = torch.device(device)
    
    # 模型注册表（必须与训练代码一致）
    MODEL_REGISTRY = {
        'unet3d': UNet3D,
        'aerb_light': AERBUNet3DLight,
        'attn_light': LightAttentionUNet3D,
        'aerb3d': AERBUNet3D,
        'attention_unet3d': AttentionUNet3D,
        'seunet3d': SEUNet3D,
        'aerb_pro': AERBPRO,
    }
    
    ModelClass = MODEL_REGISTRY.get(model_name, UNet3D)
    
    # 尝试不同的初始化参数（与训练时一致）
    # 特殊处理：AERB_pro 在训练时使用 base_channels=32（与 trainpro.py 一致）
    if model_name == 'aerb_pro':
        model_kwargs_list = [
            {'in_channels': 1, 'out_channels': 1, 'base_channels': 32},  # 与训练时一致
            {'in_channels': 1, 'base_channels': 32},
            {'in_channels': 1, 'out_channels': 1},
        ]
    else:
        model_kwargs_list = [
            {'in_channels': 1, 'out_channels': 1},
            {'in_channels': 1, 'base_channels': 16},
            {'in_channels': 1, 'out_channels': 1, 'base_channels': 16},
            {}  # 空参数
        ]
    
    model = None
    for kwargs in model_kwargs_list:
        try:
            model = ModelClass(**kwargs)
            print(f'使用参数初始化模型: {kwargs}')
            break
        except Exception as e:
            continue
    
    if model is None:
        # 最后尝试
        try:
            model = ModelClass(in_channels=1, out_channels=1)
        except:
            model = UNet3D(in_channels=1, out_channels=1)
    
    model.to(device)
    
    # ========== 加载checkpoint ==========
    if explicit_ckpt:
        checkpoint_path = Path(explicit_ckpt)
    else:
        # 自动查找策略
        def find_checkpoint(model_root, model_name):
            # 查找优先级
            search_paths = [
                model_root / model_name / 'latest',
                model_root / model_name,
                model_root
            ]
            
            # 查找的文件名模式
            patterns = [
                f'model_best_loss.pth',  # 你的训练代码保存的最佳损失模型
                f'model_best_iou.pth',   # 你的训练代码保存的最佳IoU模型
                f'model_last.pth',       # 最后模型
                f'*best*.pth',
                f'*last*.pth',
                f'*.pth'
            ]
            
            for search_dir in search_paths:
                if not search_dir.exists():
                    continue
                
                for pattern in patterns:
                    matches = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                    if matches:
                        return matches[0]
            
            return None
        
        checkpoint_path = find_checkpoint(checkpoints_root, model_name)
    
    if checkpoint_path and checkpoint_path.exists():
        print(f'加载checkpoint: {checkpoint_path}')
        
        # 检测归一化方法
        detected_norm_method = detect_normalization_method_from_checkpoint(checkpoint_path)
        if detected_norm_method == 'robust':
            use_robust_norm = True
        elif detected_norm_method == 'traditional':
            use_robust_norm = False
        print(f'使用归一化方法: {"鲁棒Z-score(MAD)" if use_robust_norm else "传统Z-score"}')
        
        try:
            state_dict = smart_load_checkpoint_dict(checkpoint_path, device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f'警告: 缺少的键: {missing_keys}')
            if unexpected_keys:
                print(f'警告: 意外的键: {unexpected_keys}')
            
            print('模型加载成功')
        except Exception as e:
            print(f'模型加载失败: {e}')
            print('尝试使用strict=False加载...')
            try:
                model.load_state_dict(state_dict, strict=False)
                print('模型以strict=False加载成功')
            except Exception as e2:
                print(f'模型加载完全失败: {e2}')
                return
    else:
        print('警告: 未找到checkpoint，使用随机初始化模型')
    
    # ========== 评估 ==========
    print('\n开始评估...')
    save_folder = save_dir if save_predictions else None
    summary, stats, micro = evaluate(model, loader_val, device, threshold=threshold,
                                     save_dir=save_folder, use_robust_norm=use_robust_norm)
    
    # ========== 打印结果 ==========
    metrics = ['dice', 'iou', 'acc', 'precision', 'recall', 'f1']
    print_mode = report_mode  # 'micro' 或 'mean'
    print('\n' + '=' * 60)
    print('评估结果')
    print('=' * 60)
    for m in metrics:
        val = micro[m] if print_mode == 'micro' else summary[m]['mean']
        print(f'{m:12s}: {val:.4f}')
    
    # 保存详细结果
    result_dir = Path('evaluation_results') / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存汇总结果
    result_file = result_dir / 'evaluation_summary.txt'
    with open(result_file, 'w') as f:
        f.write('评估汇总\n')
        f.write('=' * 50 + '\n')
        f.write(f'模型: {model_name}\n')
        f.write(f'Checkpoint: {checkpoint_path}\n')
        f.write(f'归一化方法: {"鲁棒Z-score(MAD)" if use_robust_norm else "传统Z-score"}\n')
        f.write(f'阈值: {threshold}\n')
        f.write(f'样本数: {len(ds_val)}\n\n')

        f.write(f'输出模式: {print_mode}\n')
        for m in metrics:
            val = micro[m] if print_mode == 'micro' else summary[m]['mean']
            f.write(f'{m:12s}: {val:.4f}\n')
    
    print(f'\n详细结果已保存到: {result_dir}')
    
    # ========== 可视化 ==========
    if save_predictions and save_folder:
        try:
            print('\n生成可视化...')
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            vis_dir = Path('visuals') / model_name / ts
            pred_files = sorted(Path(save_folder).glob('pred_*.npy'))
            
            if pred_files:
                visualize_predictions(pred_files, ds_val, vis_dir, max_vis=15)
                
                # ===== 保存评估参数到txt文件 =====
                config_file = vis_dir / 'evaluation_config.txt'
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write('=' * 70 + '\n')
                    f.write('模型评估配置与参数\n')
                    f.write('=' * 70 + '\n\n')
                    
                    f.write('[模型信息]\n')
                    f.write(f'模型名称: {model_name}\n')
                    f.write(f'模型路径: {checkpoint_path}\n\n')
                    
                    f.write('[数据配置]\n')
                    f.write(f'数据类型: {dat_dtype}\n')
                    f.write(f'数据形状: {dat_shape}\n')
                    f.write(f'字节序: {dat_order}\n')
                    f.write(f'验证集大小: {len(ds_val)}\n')
                    f.write(f'验证数据路径: {val_data}\n')
                    f.write(f'验证标签路径: {val_label}\n\n')
                    
                    f.write('[归一化配置]\n')
                    f.write(f'归一化方法: {"鲁棒Z-score(MAD)" if use_robust_norm else "传统Z-score"}\n')
                    if use_robust_norm:
                        f.write(f'截断范围: [-4.0, 4.0]\n')
                    else:
                        f.write(f'截断范围: [-3.0, 3.0]\n\n')
                    
                    f.write('[评估配置]\n')
                    f.write(f'阈值: {threshold}\n')
                    f.write(f'输出模式: {report_mode}\n')
                    f.write(f'Batch大小: {batch_size}\n')
                    f.write(f'设备: {device}\n\n')
                    
                    f.write('[评估结果]\n')
                    for m in metrics:
                        val = micro[m] if report_mode == 'micro' else summary[m]['mean']
                        f.write(f'{m:12s}: {val:.4f}\n')
                    
                    f.write('\n[详细统计]\n')
                    for m in metrics:
                        f.write(f'\n{m}:\n')
                        f.write(f'  均值(Mean): {summary[m]["mean"]:.4f}\n')
                        f.write(f'  标准差(Std): {summary[m]["std"]:.4f}\n')
                        f.write(f'  最小值(Min): {summary[m]["min"]:.4f}\n')
                        f.write(f'  最大值(Max): {summary[m]["max"]:.4f}\n')
                        f.write(f'  中位数(Median): {summary[m]["median"]:.4f}\n')
                    
                    f.write('\n' + '=' * 70 + '\n')
                    f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                
                print(f'评估参数已保存到: {config_file}')
            else:
                print('未找到预测文件，跳过可视化')
        except Exception as e:
            print(f'可视化失败: {e}')
    
    print('\n评估完成!')


if __name__ == '__main__':
    main()