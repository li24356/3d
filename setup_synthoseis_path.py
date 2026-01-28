"""
Synthoseis 路径配置脚本

无需安装，直接将 Synthoseis 添加到 Python 路径
"""

import sys
from pathlib import Path

# 添加 Synthoseis 目录到 Python 路径
synthoseis_path = Path(__file__).parent / 'synthoseis'

if synthoseis_path.exists():
    sys.path.insert(0, str(synthoseis_path))
    print(f"✓ 已添加 Synthoseis 路径: {synthoseis_path}")
    
    # 验证
    try:
        from datagenerator.DataGenerator import DataGenerator
        print("✓ 可以正常导入 DataGenerator")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
else:
    print(f"✗ Synthoseis 目录不存在: {synthoseis_path}")
    print("请先运行: git clone https://github.com/sede-open/synthoseis.git")
