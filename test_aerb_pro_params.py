#!/usr/bin/env python3
"""测试 AERB_PRO 参数检测逻辑"""

from pathlib import Path

# 模拟 checkpoint 路径
checkpoint_paths = [
    "checkpoints1/aerb_pro_c16/latest/model_best_iou.pth",
    "checkpoints1/aerb_pro_c32/latest/model_best_iou.pth",
    "checkpoints1/aerb_pro/latest/model_best_iou.pth",
]

for ckpt in checkpoint_paths:
    final_ckpt = Path(ckpt)
    print(f"\n测试路径: {ckpt}")
    print(f"  完整: {final_ckpt.absolute()}")
    
    # 方法1: 从父目录名推断 (如: aerb_pro_c16)
    parent_dir = final_ckpt.parent.parent.name  # latest 上一级
    print(f"  父目录: {parent_dir}")
    
    if 'c16' in parent_dir:
        inferred_base_channels = 16
    elif 'c32' in parent_dir:
        inferred_base_channels = 32
    else:
        # 备选：从完整路径推断
        ckpt_path_str = str(final_ckpt)
        if '_c16' in ckpt_path_str or 'c16' in ckpt_path_str.lower():
            inferred_base_channels = 16
        elif '_c32' in ckpt_path_str or 'c32' in ckpt_path_str.lower():
            inferred_base_channels = 32
        else:
            inferred_base_channels = 16
    
    print(f"  ✓ 推断 base_channels={inferred_base_channels}")
