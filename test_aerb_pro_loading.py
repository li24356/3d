#!/usr/bin/env python3
"""验证 AERB_PRO 模型加载整个流程"""

from pathlib import Path
import torch
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from models.AERB3d import AERBPRO

# 测试参数
checkpoints_root = Path('checkpoints1')
model_name = 'aerb_pro'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"设备: {device}")
print(f"Checkpoint根目录: {checkpoints_root}")
print(f"模型名称: {model_name}")

# 1. 查找 checkpoint
print(f"\n1. 查找 checkpoint...")

def find_preferred(dir_path: Path):
    if not dir_path.exists(): 
        print(f"   目录不存在: {dir_path}")
        return None
    for name in ('model_best_iou.pth', 'model_best.pth', f'{model_name}_best.pth'):
        if (dir_path / name).exists(): 
            return dir_path / name
    bests = sorted(dir_path.glob('*best*.pth'), key=lambda p: p.stat().st_mtime)
    if bests: return bests[-1]
    lasts = sorted(dir_path.glob('*last*.pth'), key=lambda p: p.stat().st_mtime)
    if lasts: return lasts[-1]
    return None

ckpt_to_use = None
candidate_dirs = [model_name, f'{model_name}_c16']

for sub in candidate_dirs:
    search_path = checkpoints_root / sub / 'latest'
    print(f"  搜索: {search_path}")
    ckpt_to_use = find_preferred(search_path)
    if ckpt_to_use: 
        print(f"  ✓ 找到: {ckpt_to_use}")
        break

if not ckpt_to_use:
    print("  未找到latest目录，搜索时间戳目录...")
    for sub in candidate_dirs:
        if (checkpoints_root / sub).exists():
            subdirs = sorted([d for d in (checkpoints_root/sub).iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
            for ts_dir in reversed(subdirs):
                ckpt_to_use = find_preferred(ts_dir)
                if ckpt_to_use: 
                    print(f"  ✓ 找到: {ckpt_to_use}")
                    break
            if ckpt_to_use: break

if not ckpt_to_use:
    raise FileNotFoundError(f'未找到 {model_name} 的checkpoint')

final_ckpt = Path(ckpt_to_use)
print(f"\n✓ Checkpoint: {final_ckpt}")

# 2. 推断 base_channels
print(f"\n2. 推断 base_channels...")

parent_dir = final_ckpt.parent.parent.name  # latest 上一级
print(f"   父目录: {parent_dir}")

if 'c16' in parent_dir:
    inferred_base_channels = 16
elif 'c32' in parent_dir:
    inferred_base_channels = 32
else:
    ckpt_path_str = str(final_ckpt)
    if '_c16' in ckpt_path_str or 'c16' in ckpt_path_str.lower():
        inferred_base_channels = 16
    elif '_c32' in ckpt_path_str or 'c32' in ckpt_path_str.lower():
        inferred_base_channels = 32
    else:
        inferred_base_channels = 16

print(f"   推断 base_channels={inferred_base_channels}")

# 3. 初始化模型
print(f"\n3. 初始化模型...")
model = AERBPRO(in_channels=1, out_channels=1, base_channels=inferred_base_channels)
model.to(device)
print(f"   ✓ 模型初始化成功")

# 4. 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"   参数数量: {total_params:,}")

# 5. 加载权重
print(f"\n4. 加载权重...")
try:
    checkpoint = torch.load(final_ckpt, map_location=device)
    print(f"   ✓ Checkpoint 加载成功")
    
    possible_keys = ['model_state_dict', 'model_state', 'state_dict', 'model', 'weights']
    state_dict = None
    for key in possible_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            print(f"   使用键: {key}")
            break
    
    if state_dict is None:
        state_dict = checkpoint
        print(f"   使用完整 checkpoint")
    
    # 移除 'module.' 前缀
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print(f"   移除 'module.' 前缀")
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"   ⚠ 缺失键 ({len(missing_keys)}): {missing_keys[:3]}...")
    if unexpected_keys:
        print(f"   ⚠ 意外键 ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
    
    if not missing_keys and not unexpected_keys:
        print(f"   ✓ 权重加载成功，所有键匹配")
    else:
        print(f"   ✓ 权重加载成功（存在缺失或意外的键）")
        
except Exception as e:
    print(f"   ✗ 加载失败: {e}")
    raise

print(f"\n✅ AERBPRO 模型加载完成！")
print(f"   base_channels: {inferred_base_channels}")
print(f"   参数数量: {total_params:,}")
print(f"   设备: {device}")
