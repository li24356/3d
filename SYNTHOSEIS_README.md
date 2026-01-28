# Synthoseis 数据生成指南

## 快速开始

已生成了 3 个演示样本，你可以在 `demo_output/` 目录中查看：
- `sample_000_slices.png`
- `sample_001_slices.png`
- `sample_002_slices.png`

每张图显示地震数据和断层标签在三个方向上的切片：
- 上排：地震数据（灰度）
- 下排：断层标签（颜色）
- 左列：Z方向切片（水平层）
- 中列：Y方向切片（纵断面）
- 右列：X方向切片（纵断面）

## 安装 Synthoseis

### 方法 1: 使用 pip（推荐）
```bash
pip install synthoseis
```

### 方法 2: 从 GitHub 克隆
```bash
git clone https://github.com/sede-open/synthoseis.git
cd synthoseis
pip install -e .
```

### 依赖项
- Python >= 3.8
- numpy
- scipy
- matplotlib
- scikit-image

完整列表参见 `environment.yml` 或 `pyproject.toml`

## 生成完整数据集

安装 Synthoseis 后，运行：
```bash
python generate_synthoseis_data.py
```

这将生成：
- **训练集**：200 个 128×128×128 的样本 → `synthetic_data_v2/train/`
- **验证集**：20 个 128×128×128 的样本 → `synthetic_data_v2/prediction/`

每个样本包含：
- `seis_XXXXXX.npy`：地震数据（float32）
- `fault_XXXXXX.npy`：断层标签（uint8，0=非断层，1=断层）

## Synthoseis 配置说明

生成脚本中的主要参数：

### 数据尺寸
- `cube_shape = [128, 128, 128]` - 输出体积（X, Y, Z）

### 断层参数
- `min_number_faults = 1` - 最少断层数
- `max_number_faults = 3` - 最多断层数
- `dip_factor_max = 0.5` - 断层倾角系数

### 地层参数
- `thickness_min = 4` - 最小地层厚度（采样点）
- `thickness_max = 20` - 最大地层厚度
- `sand_layer_fraction = 0.3` - 砂层比例

### 地震参数
- `incident_angles = [0, 30]` - 入射角（两个角度）
- `digi = 4` - 采样率（毫秒）
- `signal_to_noise_ratio_db = [15, 20, 25]` - 信噪比范围

### 频率
- `bandwidth_low = [8, 12]` - 低频截止范围（Hz）
- `bandwidth_high = [80, 100]` - 高频截止范围（Hz）

## 自定义数据生成

编辑 `generate_synthoseis_data.py` 中的 `SYNTHOSEIS_CONFIG` 字典来调整参数：

```python
SYNTHOSEIS_CONFIG = {
    "cube_shape": [256, 256, 256],  # 改为 256×256×256
    "min_number_faults": 2,          # 至少 2 条断层
    "max_number_faults": 5,          # 最多 5 条断层
    # ... 其他参数
}

NUM_TRAIN = 500  # 改为 500 个训练样本
NUM_VAL = 50     # 改为 50 个验证样本
```

## 文件结构

生成完成后的目录结构：
```
synthetic_data_v2/
├── train/
│   ├── seis/
│   │   ├── 000000.npy
│   │   ├── 000001.npy
│   │   └── ...（200个文件）
│   └── fault/
│       ├── 000000.npy
│       ├── 000001.npy
│       └── ...（200个文件）
├── prediction/  （验证集）
│   ├── seis/
│   │   ├── 000000.npy
│   │   └── ...（20个文件）
│   └── fault/
│       ├── 000000.npy
│       └── ...（20个文件）
└── synthoseis_work/  （临时文件）
```

## 与训练脚本集成

生成的数据可直接用于训练：
```bash
python train.py
```

脚本会自动从 `synthetic_data_v2/train/` 和 `synthetic_data_v2/prediction/` 加载数据。

## 性能建议

- **快速生成**：使用 128×128×128，NUM_TRAIN=10，NUM_VAL=2
- **标准配置**：使用 128×128×128，NUM_TRAIN=200，NUM_VAL=20
- **大规模**：使用 256×256×256，NUM_TRAIN=500，NUM_VAL=50（需要更多显存）

## 参考文献

Merrifield et al. (2022). "Synthetic seismic data for training deep learning networks". 
*Interpretation*, 10(3), SE31-SE39.
https://doi.org/10.1190/INT-2021-0193.1

## 更多信息

- GitHub: https://github.com/sede-open/synthoseis
- 文档: https://sede-open.github.io/synthoseis/
