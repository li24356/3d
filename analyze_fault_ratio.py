import numpy as np
import os
from pathlib import Path

# 设置fault文件夹路径
fault_dir = Path(r"h:\3d\train\fault")

# 统计变量
total_zeros = 0
total_ones = 0
total_samples = 0
files_with_faults = 0  # 包含断层的文件数量
files_without_faults = 0  # 不包含断层的文件数量

# 遍历所有.dat文件
dat_files = sorted(fault_dir.glob("*.dat"))
print(f"找到 {len(dat_files)} 个文件")

for dat_file in dat_files:
    try:
        # 读取数据
        data = np.fromfile(dat_file, dtype=np.float32)
        
        # 统计0和1的数量
        zeros = np.sum(data == 0)
        ones = np.sum(data == 1)
        
        total_zeros += zeros
        total_ones += ones
        total_samples += len(data)
        
        # 检查是否包含断层
        if ones > 0:
            files_with_faults += 1
        else:
            files_without_faults += 1
            print(f"  ⚠️ {dat_file.name} 不包含断层（全为0）")
        
        # 打印前几个文件的统计信息
        if dat_files.index(dat_file) < 5:
            print(f"\n{dat_file.name}:")
            print(f"  总元素数: {len(data)}")
            print(f"  0的数量: {zeros} ({zeros/len(data)*100:.2f}%)")
            print(f"  1的数量: {ones} ({ones/len(data)*100:.2f}%)")
            print(f"  数据形状信息: min={data.min()}, max={data.max()}")
    except Exception as e:
        print(f"读取 {dat_file.name} 时出错: {e}")

# 打印总体统计
print("\n" + "="*60)
print("总体统计:")
print(f"总文件数: {len(dat_files)}")
print(f"包含断层的文件: {files_with_faults} ({files_with_faults/len(dat_files)*100:.1f}%)")
print(f"不包含断层的文件: {files_without_faults} ({files_without_faults/len(dat_files)*100:.1f}%)")
print(f"总元素数: {total_samples}")
print(f"0的总数量: {total_zeros} ({total_zeros/total_samples*100:.2f}%)")
print(f"1的总数量: {total_ones} ({total_ones/total_samples*100:.2f}%)")
if total_ones > 0:
    print(f"0:1 比例 = {total_zeros}:{total_ones} ≈ {total_zeros/total_ones:.2f}:1")
print("="*60)
