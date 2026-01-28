# ⚡ Synthoseis 快速使用指南

## 📦 安装状态

✓ Synthoseis 已克隆到 `h:\3d\synthoseis\`  
✓ 可以无需正式安装直接使用

## 🚀 快速开始

### 只生成演示数据（推荐首先尝试）
```bash
cd h:\3d
python generate_data_advanced.py --demo-only
```

这会生成 3 个演示样本，可以在 `demo_output/` 中查看切片图。

### 生成完整数据集（200 训练 + 20 验证）
```bash
cd h:\3d
python generate_data_advanced.py
```

## 📊 输出文件结构

完成后的目录结构：
```
synthetic_data_v2/
├── train/
│   ├── seis/           # 地震数据
│   │   ├── 000000.npy  # 128×128×128 地震数据
│   │   ├── 000001.npy
│   │   └── ... (200个)
│   └── fault/          # 断层标签
│       ├── 000000.npy  # 128×128×128 二值标签
│       ├── 000001.npy
│       └── ... (200个)
├── prediction/         # 验证集
│   ├── seis/
│   │   └── ... (20个)
│   └── fault/
│       └── ... (20个)
```

## 🔧 自定义参数

### 改变数据尺寸
```bash
# 生成 256×256×256 的数据
python generate_data_advanced.py --cube-size 256 256 256
```

### 改变样本数量
```bash
# 生成 500 训练 + 100 验证样本
python generate_data_advanced.py --num-train 500 --num-val 100
```

### 组合参数
```bash
# 生成 300 个 256×256×256 的训练样本
python generate_data_advanced.py --num-train 300 --cube-size 256 256 256 --demo-only
```

## 📁 数据格式

### 地震数据 (seis_XXXXXX.npy)
- 格式：float32
- 形状：(Z, Y, X) = (128, 128, 128)
- 范围：通常 -1.0 ~ 1.0
- 代表：地震反射强度

### 断层标签 (fault_XXXXXX.npy)
- 格式：uint8
- 形状：(Z, Y, X) = (128, 128, 128)
- 值：0（非断层）或 1（断层）
- 用途：训练标签

## 🔍 验证数据

检查数据是否正确加载：
```python
import numpy as np

# 加载一个样本
seis = np.load('synthetic_data_v2/train/seis/000000.npy')
fault = np.load('synthetic_data_v2/train/fault/000000.npy')

print(f"地震数据形状: {seis.shape}, 类型: {seis.dtype}")
print(f"地震数据范围: {seis.min():.3f} ~ {seis.max():.3f}")
print(f"\n断层标签形状: {fault.shape}, 类型: {fault.dtype}")
print(f"断层像素比: {fault.mean()*100:.2f}%")
```

## 🎯 与训练脚本集成

生成的数据已准备好用于训练：

```bash
# 修改 dataloader.py 或 train.py 中的数据路径
# 将数据路径指向 synthetic_data_v2/

# 然后运行训练
python train.py
```

## ⚠️ 常见问题

### Q: 生成速度很慢？
A: Synthoseis 生成真实地震数据会很耗时。
- 首先用 `--demo-only` 快速生成演示数据
- 可以减少 `--num-train` 的数量来加快速度

### Q: 内存不足？
A: 降低 `--cube-size` 或 `--num-train`
```bash
python generate_data_advanced.py --cube-size 64 64 64 --num-train 50
```

### Q: 如何修改地层/断层参数？
A: 编辑 `generate_data_advanced.py` 中的 `Config.SYNTHOSEIS_CONFIG` 字典

### Q: 能否生成更大的数据集？
A: 可以，但会很慢。建议分批生成。

## 📚 参考资源

- GitHub: https://github.com/sede-open/synthoseis
- 文档: https://sede-open.github.io/synthoseis/
- 论文: https://doi.org/10.1190/INT-2021-0193.1

## 🎬 下一步

1. **生成演示数据**
   ```bash
   python generate_data_advanced.py --demo-only
   ```

2. **查看 demo_output 中的 PNG 图片**

3. **如果满意，生成完整数据集**
   ```bash
   python generate_data_advanced.py
   ```

4. **开始训练**
   ```bash
   python train.py
   ```
