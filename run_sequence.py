import time
from datetime import datetime
import torch
from pathlib import Path
from typing import Optional

import train  # 导入你项目中的 train.py 模块（确保与 run_sequence.py 在同一工作目录）

# 顺序列表：(model_name, optional model_tag)
SEQUENCE = [
    ("attn_light", "attn_light"),
    ("aerb_light", "aerb_light"),
]

def run_model_once(model_name: str, model_tag: Optional[str] = None):
    print("="*60)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  开始训练模型: {model_name}")
    # 修改 train 模块中的 MODEL_CONFIG
    train.MODEL_CONFIG['model_name'] = model_name
    if model_tag is not None:
        train.MODEL_CONFIG['model_tag'] = model_tag
    else:
        train.MODEL_CONFIG['model_tag'] = None

    # 可选：在这里可以修改 train.py 顶部的其他配置（如果需要）
    # 例如：train.MODEL_CONFIG['base_channels'] = 16

    try:
        train.main()
    except Exception as e:
        print(f"训练模型 {model_name} 时发生异常: {e}")
        raise
    finally:
        # 清理显存与短暂休眠，确保下一个模型能顺利启动
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  结束训练模型: {model_name}")
        print("="*60 + "\n")

def main():
    # 逐个运行序列中的模型
    for model_name, model_tag in SEQUENCE:
        run_model_once(model_name, model_tag)

if __name__ == "__main__":
    main()