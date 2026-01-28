import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from tqdm import tqdm

# 引入你的模型
from models.attention_unet3d import LightAttentionUNet3D

# ================= 配置区域 =================
# 1. 输入数据
DATA_PATH = Path(r"2020Z205_3D_PSTM_TIME_mini_400_2600ms.npy")

# 2. 模型配置
MODEL_NAME = 'attn_light'
# 设为 None 自动查找，或填入路径
CHECKPOINT_PATH = None         
CHECKPOINTS_ROOT = Path('checkpoints5')

# 3. 输出目录
OUTPUT_DIR = Path('permutation_test_results')

# 4. 测试块大小 (切取中心部分)
CROP_SIZE = (128, 128, 128)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===========================================

def find_checkpoint(root, model_name):
    """支持 *_c16/_c32 等目录的 checkpoint 搜索"""
    candidate_names = [model_name, f"{model_name}_c16", f"{model_name}_c32", f"{model_name}_c64"]
    preferred_files = ['model_best_iou.pth', 'model_best_loss.pth', 'model_best.pth', 'model_last.pth']
    
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


def load_model(ckpt_path, device):
    print(f"加载模型权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 智能提取 state_dict (支持多种包装格式)
    possible_keys = ['model_state', 'model_state_dict', 'state_dict', 'model']
    state_dict = None
    
    if isinstance(checkpoint, dict):
        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"  从 checkpoint['{key}'] 提取权重")
                break
        if state_dict is None:
            # 如果没找到标准键，检查是否整个字典就是 state_dict
            # (通过检查是否包含模型参数的键)
            if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                state_dict = checkpoint
                print("  直接使用 checkpoint 作为 state_dict")
            else:
                print("  ⚠️ 无法识别 checkpoint 格式，尝试使用整个字典")
                state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 去除 module. 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 尝试初始化 LightAttentionUNet3D (不需要 base_channels 参数)
    model = LightAttentionUNet3D(in_channels=1, out_channels=1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"  警告: 缺少的键: {missing[:5]}..." if len(missing) > 5 else f"  警告: 缺少的键: {missing}")
    if unexpected:
        print(f"  警告: 意外的键: {unexpected[:5]}..." if len(unexpected) > 5 else f"  警告: 意外的键: {unexpected}")
    if not missing and not unexpected:
        print("  -> 权重完全匹配")
        
    model.to(device)
    model.eval()
    return model

def robust_normalize(x):
    """GPU版鲁棒归一化"""
    flat = x.view(1, -1)
    median = torch.median(flat)
    mad = torch.median(torch.abs(flat - median))
    scale = mad * 1.4826 if mad > 1e-6 else 1.0
    return torch.clamp((x - median) / scale, -4.0, 4.0)

def visualize(save_path, perm, vol_in, vol_pred):
    """画图：输入切片 vs 预测切片"""
    # vol_in/vol_pred 已经是转换后的视角，我们直接展示中心切片
    D, H, W = vol_in.shape
    cx, cy, cz = W//2, H//2, D//2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：输入数据
    axes[0,0].imshow(vol_in[cz,:,:], cmap='seismic')
    axes[0,0].set_title(f"Input Axis-0 (dim={D})")
    axes[0,1].imshow(vol_in[:,cy,:], cmap='seismic')
    axes[0,1].set_title(f"Input Axis-1 (dim={H})")
    axes[0,2].imshow(vol_in[:,:,cx], cmap='seismic')
    axes[0,2].set_title(f"Input Axis-2 (dim={W})")
    
    # 第二行：预测结果
    axes[1,0].imshow(vol_pred[cz,:,:], cmap='gray', vmin=0, vmax=1)
    axes[1,0].set_title("Pred Axis-0")
    axes[1,1].imshow(vol_pred[:,cy,:], cmap='gray', vmin=0, vmax=1)
    axes[1,1].set_title("Pred Axis-1")
    axes[1,2].imshow(vol_pred[:,:,cx], cmap='gray', vmin=0, vmax=1)
    axes[1,2].set_title("Pred Axis-2")
    
    perm_str = str(perm).replace(' ', '')
    plt.suptitle(f"Permutation: {perm_str}\n(Input Shape: {vol_in.shape})", fontsize=16, color='red')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取全量数据
    print(f"读取数据: {DATA_PATH} ...")
    full_data = np.load(DATA_PATH).reshape(501, 601, 1101, order='C') # 假设数据已经是 (D, H, W)
    print(f"原始数据形状: {full_data.shape} (设为 dim 0, 1, 2)")
    
    # 2. 截取中心块 (避免内存溢出，加快速度)
    d0, d1, d2 = full_data.shape
    c0, c1, c2 = d0//2, d1//2, d2//2
    s0, s1, s2 = CROP_SIZE
    
    # 确保切片不越界
    crop = full_data[
        c0-s0//2 : c0+s0//2,
        c1-s1//2 : c1+s1//2,
        c2-s2//2 : c2+s2//2
    ].astype(np.float32)
    print(f"截取中心块用于测试: {crop.shape}")
    
    # 转 Tensor 并归一化
    crop_t = torch.from_numpy(crop).to(DEVICE)
    crop_norm = robust_normalize(crop_t) # (D, H, W)
    
    # 3. 准备模型
    if CHECKPOINT_PATH:
        ckpt_path = Path(CHECKPOINT_PATH)
    else:
        # 使用增强版查找函数 (支持 *_c16/_c32 等)
        ckpt_path = find_checkpoint(CHECKPOINTS_ROOT, MODEL_NAME)
    
    if not ckpt_path:
        print("❌ 未找到权重，请手动设置 CHECKPOINT_PATH")
        return
        
    model = load_model(ckpt_path, DEVICE)
    
    # 4. 暴力穷举所有排列
    # 0,1,2 代表原始数据的三个维度
    permutations = list(itertools.permutations([0, 1, 2])) 
    
    print(f"\n开始测试 {len(permutations)} 种排列方式...")
    
    with torch.no_grad():
        for i, perm in enumerate(permutations):
            print(f"[{i+1}/{6}] 测试排列: {perm} ...")
            
            # 1. 维度变换
            # permute 输入: (D, H, W) -> Permuted
            input_tensor = crop_norm.permute(*perm)
            
            # 添加 Batch 和 Channel 维: (1, 1, D', H', W')
            model_input = input_tensor.unsqueeze(0).unsqueeze(0)
            
            # 2. 推理
            logits = model(model_input)
            probs = torch.sigmoid(logits)
            
            # 3. 获取结果 (去除 B, C)
            pred_vol = probs[0, 0].cpu().numpy()
            input_vol = input_tensor.cpu().numpy()
            
            # 4. 保存图片
            save_name = OUTPUT_DIR / f"perm_{perm[0]}{perm[1]}{perm[2]}.png"
            visualize(save_name, perm, input_vol, pred_vol)
            
    print(f"\n✅ 测试完成！结果保存在: {OUTPUT_DIR}")
    print("请打开文件夹，寻找那张【断层显示为清晰、连续线条（而非杂乱噪点或层位）】的图片。")
    print("图片标题上的 Permutation (例如 2,0,1) 就是你需要填入 predict_large.py 的转置参数！")

if __name__ == "__main__":
    main()