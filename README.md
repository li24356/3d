# 简单 3D U-Net 与训练脚本

此仓库添加了一个最小的 3D U-Net (`models/unet3d.py`)、数据加载器 (`dataloader.py`) 和训练脚本 (`train.py`)，用于处理 128x128x128 的体数据。

目录约定（与您的描述一致）：
- `train/seis` : 训练数据（体数据，NumPy `.npy` 或 `.npz`）
- `train/fault`: 训练标签（同样为 `.npy` 或 `.npz`）
- `prediction/seis`: 验证/预测数据
- `prediction/fault`: 验证标签

用法示例（在 PowerShell 下）:

```powershell
# 安装依赖（请在合适的 Python 虚拟环境中运行）
python -m pip install -r requirements.txt

# 使用 train/ 进行训练，prediction/ 作为验证
python train.py --root . --epochs 50 --batch-size 1 --lr 1e-4 --save-dir checkpoints
```

注意：
- 数据加载器默认按文件名（去掉扩展名）配对 `seis` 与 `fault` 下的文件。
- 期望输入体积大小为 128x128x128；如果大小不同，代码会进行中心裁剪或常数填充以适配。
- 标签应为二值或概率体（训练中使用 `BCEWithLogitsLoss` + DiceLoss）。

支持 `.dat` 原始二进制文件
--------------------------------
如果你的数据是原始二进制 `.dat` 文件，数据加载器现在支持直接读取，但你需要在创建数据集时指定二进制数据的 dtype 和体积 shape。

示例：你的 `.dat` 文件保存的是连续的 `float32` 值，体积尺寸为 `128 x 128 x 128`，可以这样在代码中使用：

```python
from dataloader import VolumeDataset

ds = VolumeDataset('train/seis', 'train/fault', dat_dtype='float32', dat_shape=(128,128,128))
```

注意：
- `dat_dtype` 应为 NumPy 能识别的 dtype 字符串（如 `float32`, `int16` 等）。
- `dat_shape` 必须与磁盘上 `.dat` 文件的体素排列一致（默认按行主序 `order='C'` 读取）。
- 如果 `.dat` 文件的元素数量与 `dat_shape` 给定的大小不匹配，加载会报错以提示检查参数。

如果你希望我做下面任意一项，请告诉我：
- 把模型改为多通道输入/输出（例如输入 3 通道）
- 添加更强的数据增强（随机翻转、旋转）
- 用 NIfTI (`.nii`) 或其他格式支持数据加载
- 运行一次快速 smoke-test（需要一些示例 `.npy` 文件）
