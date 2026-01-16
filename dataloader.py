import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # 引入进度条

def _list_volume_files(folder: str) -> List[str]:
    exts = ('.npy', '.npz', '.dat')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files

def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

class VolumeDataset(Dataset):
    """
    带 RAM 缓存的 Dataset。
    初始化时会将所有数据预处理并加载到内存中，极大提升训练速度。
    """
    def __init__(self, data_dir: str, label_dir: str, patch_size=128, normalize=True,
                 dat_dtype: Optional[str] = None, dat_shape: Optional[Tuple[int, int, int]] = None,
                 dat_order: str = 'C'):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.normalize = normalize
        self.dat_dtype = dat_dtype
        self.dat_shape = dat_shape
        self.dat_order = dat_order

        # 1. 扫描文件并配对
        data_files = _list_volume_files(data_dir) if os.path.isdir(data_dir) else []
        label_files = _list_volume_files(label_dir) if os.path.isdir(label_dir) else []

        data_map = { _basename_no_ext(p): p for p in data_files }
        label_map = { _basename_no_ext(p): p for p in label_files }

        keys = sorted(set(data_map.keys()) & set(label_map.keys()))
        self.pairs: List[Tuple[str, str]] = [(data_map[k], label_map[k]) for k in keys]

        if len(self.pairs) == 0:
            raise RuntimeError(f'No paired volumes found in {data_dir} and {label_dir}')

        # 2. 【核心修改】RAM Cache 预加载
        # 我们在这里就把数据读好、处理好、转成 Tensor，存进列表里
        print(f"正在将 {len(self.pairs)} 对数据预加载到内存 (RAM Cache)...")
        self.cache = []

        for data_path, label_path in tqdm(self.pairs, desc="Pre-loading Data"):
            # 调用原本的处理逻辑
            data_t, label_t = self._process_one_sample(data_path, label_path)
            self.cache.append((data_t, label_t))
        
        print("✅ 数据预加载完成！")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx: int):
        # 训练时直接从内存取，极快
        return self.cache[idx]

    # =========================================================================
    # 下面保留你原有的读取和处理逻辑，改为内部辅助函数
    # =========================================================================

    def _process_one_sample(self, data_path, label_path):
        """读取并处理单个样本（整合了你原代码 __getitem__ 的逻辑）"""
        data = self._load(data_path)
        label = self._load(label_path)

        # ensure shapes: (Z,Y,X)
        if data.ndim == 4:
            data = data[0]
        if label.ndim == 4:
            label = label[0]

        data = data.astype('float32')
        if self.normalize:
            m = data.mean()
            s = data.std()
            if s > 0:
                data = (data - m) / s

        label = label.astype('float32')

        # 使用你的 fix_size 逻辑
        data = self._fix_size(data)
        label = self._fix_size(label)

        # 增加 Channel 维度
        data = np.expand_dims(data, 0)
        label = np.expand_dims(label, 0)

        data_t = torch.from_numpy(data)
        label_t = torch.from_numpy(label)

        return data_t, label_t

    def _fix_size(self, vol):
        """你原来的 padding/cropping 逻辑"""
        z, y, x = vol.shape
        sz = self.patch_size
        pad = [0, 0, 0, 0, 0, 0]
        if x < sz:
            dx = sz - x
            pad[0] = dx // 2
            pad[1] = dx - pad[0]
        if y < sz:
            dy = sz - y
            pad[2] = dy // 2
            pad[3] = dy - pad[2]
        if z < sz:
            dz = sz - z
            pad[4] = dz // 2
            pad[5] = dz - pad[4]
        if any(pad):
            vol = np.pad(vol, ((pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])), mode='constant')
        z, y, x = vol.shape
        startz = max(0, (z - sz) // 2)
        starty = max(0, (y - sz) // 2)
        startx = max(0, (x - sz) // 2)
        vol = vol[startz:startz + sz, starty:starty + sz, startx:startx + sz]
        return vol

    def _load(self, path: str) -> np.ndarray:
        if path.lower().endswith('.dat'):
            return self._load_dat(path)
        else:
            return self._load_np(path)

    def _load_dat(self, path: str) -> np.ndarray:
        if self.dat_dtype is None or self.dat_shape is None:
            raise RuntimeError('To load .dat files you must provide dat_dtype and dat_shape to VolumeDataset')
        dtype = np.dtype(self.dat_dtype)
        arr = np.fromfile(path, dtype=dtype)
        expected = int(np.prod(self.dat_shape))
        if arr.size != expected:
            raise RuntimeError(f'.dat file {path} has {arr.size} elements but expected {expected} for shape {self.dat_shape}')
        arr = arr.reshape(self.dat_shape, order=self.dat_order)
        return arr

    def _load_np(self, path: str) -> np.ndarray:
        if path.lower().endswith('.npz'):
            data = np.load(path)
            arr = None
            for v in data.files:
                arr = data[v]
                break
            if arr is None:
                raise RuntimeError(f'Empty npz: {path}')
            return arr
        else:
            return np.load(path)

if __name__ == '__main__':
    try:
        # 简单测试逻辑
        # 确保你在目录下有相应文件夹，否则会报错
        ds = VolumeDataset('train/seis', 'train/fault', dat_dtype='float32', dat_shape=(128,128,128))
        print('Total pairs loaded:', len(ds))
        x, y = ds[0]
        print('Sample shape:', x.shape, y.shape)
    except Exception as e:
        print('Test skipped or failed (normal if no data):', e)