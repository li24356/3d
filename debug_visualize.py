import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入你的项目模块
from dataloader import VolumeDataset
from models.unet3d import UNet3D
from models.AERB3d import AERBUNet3D, AERBUNet3DLight
from models.seunet3d import SEUNet3D
from models.AERB_pro import AERBPRO
from models.attention_unet3d import AttentionUNet3D, LightAttentionUNet3D

# ================= 配置区域 =================
# 1. 模型与路径
MODEL_NAME = 'attn_light'        # 模型名称
CHECKPOINT_PATH = None         # 设为 None 自动查找，或填入具体路径 str
CHECKPOINTS_ROOT = Path('checkpoints5')  # 查找 checkpoint 的根目录

# 2. 数据路径 (请确保这些目录下有 .npy 或 .dat 文件)
DATA_DIR = Path('synthetic_data_v2/prediction/seis')
LABEL_DIR = Path('synthetic_data_v2/prediction/fault')

# 3. 输出目录
VIS_OUTPUT_DIR = Path('debug_visuals')

# 4. 参数
PATCH_SIZE = (128, 128, 128)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORM_METHOD = 'robust'         # 'robust' 或 'traditional'
MAX_SAMPLES = 10               # 只可视化前 10 个样本
THRESHOLD = 0.5                # 二值化阈值
# ===========================================

def robust_normalize(x):
    """鲁棒归一化 (与训练保持一致)"""
    x = x.float()
    # (B, C, D, H, W) -> (B, -1)
    flat = x.view(x.size(0), -1)
    
    # 计算中位数和MAD
    median = torch.median(flat, dim=1, keepdim=True)[0]
    abs_diff = torch.abs(flat - median)
    mad = torch.median(abs_diff, dim=1, keepdim=True)[0]
    
    scale = torch.where(mad > 1e-6, mad * 1.4826, torch.ones_like(mad))
    
    # 归一化并截断
    x_norm = (flat - median) / scale
    x_norm = torch.clamp(x_norm, -4.0, 4.0)
    
    return x_norm.view_as(x)

def smart_load_checkpoint_dict(ck_path, device):
    """对齐 evaluate.py 的智能加载：处理各种 state_dict 包装并去前缀"""
    ck = torch.load(str(ck_path), map_location=device)
    possible_keys = ['model_state', 'state_dict', 'model']
    if isinstance(ck, dict):
        st = None
        for key in possible_keys:
            if key in ck:
                st = ck[key]
                break
        if st is None:
            st = ck
    else:
        st = ck

    if isinstance(st, dict) and 'model_state' in st:
        st = st['model_state']

    new_state = {}
    for k, v in st.items():
        for prefix in ['module.', 'model.', '_orig_mod.']:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        new_state[k] = v
    return new_state


def find_checkpoint(root, model_name):
    """与 evaluate.py 对齐的 checkpoint 搜索，支持 *_c16/_c32 等目录"""
    candidate_names = [model_name, f"{model_name}_c16", f"{model_name}_c32", f"{model_name}_c64"]
    preferred_files = ['model_best_iou.pth', 'model_best_loss.pth', 'model_best.pth', 'model_last.pth']

    # 先查找 latest 目录，然后根目录
    for cname in candidate_names:
        for d in [root / cname / 'latest', root / cname]:
            if not d.exists():
                continue
            for fname in preferred_files:
                p = d / fname
                if p.exists():
                    return p
            # 回退：按修改时间排序的 *.pth
            matches = sorted(d.glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]
    return None


def load_model(model_name, ckpt_path, device):
    print(f"正在加载模型: {model_name} ...")

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

    # 尝试与训练一致的初始化参数
    if model_name == 'aerb_pro':
        model_kwargs_list = [
            {'in_channels': 1, 'out_channels': 1, 'base_channels': 16},
            {'in_channels': 1, 'out_channels': 1, 'base_channels': 32},
            {'in_channels': 1, 'out_channels': 1},
        ]
    else:
        model_kwargs_list = [
            {'in_channels': 1, 'out_channels': 1},
            {'in_channels': 1, 'base_channels': 16},
            {'in_channels': 1, 'out_channels': 1, 'base_channels': 16},
            {},
        ]

    model = None
    for kwargs in model_kwargs_list:
        try:
            model = ModelClass(**kwargs)
            print(f"  使用参数初始化模型: {kwargs}")
            break
        except Exception:
            continue

    if model is None:
        # 最后兜底
        model = ModelClass(in_channels=1, out_channels=1)
        print("  使用兜底初始化参数: in_channels=1, out_channels=1")

    # 如果是 AERBPRO，根据 ckpt 路径推断 base_channels 并重建
    if model_name == 'aerb_pro' and ckpt_path:
        s = str(ckpt_path).lower()
        inferred = 16
        if 'c32' in s or '_c32' in s:
            inferred = 32
        elif 'c64' in s or '_c64' in s:
            inferred = 64
        if hasattr(model, 'base_channels') and getattr(model, 'base_channels', None) != inferred:
            print(f"  根据路径推断 base_channels={inferred}，重新初始化模型")
            model = AERBPRO(in_channels=1, out_channels=1, base_channels=inferred)

    # 加载权重
    if ckpt_path and Path(ckpt_path).exists():
        print(f"  加载权重: {ckpt_path}")
        state_dict = smart_load_checkpoint_dict(ckpt_path, device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  警告: 缺少的键: {missing}")
        if unexpected:
            print(f"  警告: 意外的键: {unexpected}")
        if not missing and not unexpected:
            print("  权重加载状态: 完全匹配")

    model.to(device)
    model.eval()
    return model

def visualize_sample(save_path, idx, input_vol, label_vol, pred_vol):
    """生成详细的三视图对比图"""
    # input_vol, label_vol, pred_vol 都是 (D, H, W) 的 numpy 数组
    D, H, W = input_vol.shape
    
    # 取中心切片
    cx, cy, cz = W // 2, H // 2, D // 2
    
    # 创建画布：3行 (Input, Label, Pred) x 3列 (Z-slice, Y-slice, X-slice)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # 行标题
    rows = ['Seismic Input', 'GT Label', 'Prediction']
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large', labelpad=5)

    # --- 第一行：Input (Seismic) ---
    # Z切片 (Top view)
    axes[0, 0].imshow(input_vol[cz, :, :], cmap='gray')
    axes[0, 0].set_title(f'Z-slice (idx={cz})')
    # Y切片 (Front view)
    axes[0, 1].imshow(input_vol[:, cy, :], cmap='gray')
    axes[0, 1].set_title(f'Y-slice (idx={cy})')
    # X切片 (Side view)
    axes[0, 2].imshow(input_vol[:, :, cx], cmap='gray')
    axes[0, 2].set_title(f'X-slice (idx={cx})')

    # --- 第二行：Label (Ground Truth) ---
    axes[1, 0].imshow(label_vol[cz, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].imshow(label_vol[:, cy, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].imshow(label_vol[:, :, cx], cmap='gray', vmin=0, vmax=1)

    # --- 第三行：Prediction ---
    axes[2, 0].imshow(pred_vol[cz, :, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 1].imshow(pred_vol[:, cy, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 2].imshow(pred_vol[:, :, cx], cmap='gray', vmin=0, vmax=1)

    # 统一美化
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.suptitle(f'Sample {idx} Visualization\nShape: {input_vol.shape}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. 准备路径
    if not DATA_DIR.exists() or not LABEL_DIR.exists():
        print(f"错误: 数据目录不存在!")
        print(f"  Seis: {DATA_DIR}")
        print(f"  Fault: {LABEL_DIR}")
        return

    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. 准备数据
    # 注意：这里使用 dataloader.py 里的 VolumeDataset
    # 它会自动读取 patch (128,128,128)
    ds = VolumeDataset(str(DATA_DIR), str(LABEL_DIR), 
                      dat_dtype='float32', dat_shape=PATCH_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"验证集样本数: {len(ds)}")

    # 3. 准备模型
    ckpt = CHECKPOINT_PATH
    if ckpt is None:
        ckpt = find_checkpoint(CHECKPOINTS_ROOT, MODEL_NAME)
    
    if ckpt is None:
        print("错误: 未找到模型 Checkpoint，请手动指定路径！")
        return
        
    model = load_model(MODEL_NAME, str(ckpt), DEVICE)

    # 4. 循环可视化
    print(f"\n开始生成可视化 (前 {MAX_SAMPLES} 个样本)...")
    print(f"图片保存至: {VIS_OUTPUT_DIR}")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            if i >= MAX_SAMPLES: break
            
            x = x.to(DEVICE)
            
            # 归一化
            if NORM_METHOD == 'robust':
                x_norm = robust_normalize(x)
            else:
                x_norm = x # 简单起见，或者你可以加传统归一化
            
            # 推理
            logits = model(x_norm)
            probs = torch.sigmoid(logits)
            
            # 转 numpy
            # (1, 1, D, H, W) -> (D, H, W)
            vol_in = x[0, 0].cpu().numpy()
            vol_lbl = y[0, 0].cpu().numpy()
            vol_pred = probs[0, 0].cpu().numpy()
            
            # 保存图
            save_name = VIS_OUTPUT_DIR / f"vis_sample_{i:03d}.png"
            visualize_sample(save_name, i, vol_in, vol_lbl, vol_pred)
            
    print("\n完成！请打开文件夹查看图片。")
    print("------------------------------------------------")
    print("【排错指南】")
    print("1. 查看 'GT Label' (第二行):")
    print("   - 如果断层线在 'Z-slice' 里看起来像长线条，而在 'X/Y-slice' 里是点或短线 -> 说明断层是竖直的（正常）。")
    print("   - 如果断层线在 'X/Y-slice' 里是大片连通的，而 Z-slice 里没有 -> 说明断层可能是横向层状的（数据可能转置了）。")
    print("2. 查看 'Prediction' (第三行):")
    print("   - 如果 Pred 是横向的（层状），但 GT 是竖向的 -> 模型学错了，或者模型输入的维度定义和训练数据不一致。")
    print("------------------------------------------------")

if __name__ == '__main__':
    main()