import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class SeismicGenerator:
    def __init__(self, shape=(128, 128, 128), dt=0.002):
        self.shape = shape
        self.n1, self.n2, self.n3 = shape
        self.dt = dt

    def ricker_wavelet(self, f, length=0.1):
        t = np.arange(-length/2, length/2, self.dt)
        y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
        return y

    def generate_reflectivity(self):
        """【修改】生成稀疏反射系数（模拟真实层状地层）"""
        # 初始化全0
        r = np.zeros(self.n1)
        
        # 随机决定有多少个反射层 (比如每 10-20 个采样点一层)
        min_layer_thickness = 8
        num_layers = self.n1 // np.random.randint(min_layer_thickness, min_layer_thickness * 2)
        
        # 随机选择反射层的位置
        # 使用 replace=False 保证位置不重复
        indices = np.random.choice(np.arange(5, self.n1 - 5), num_layers, replace=False)
        
        # 给这些位置赋值随机反射强度 (-1 到 1 之间，避开 0)
        # 模拟强反射界面
        amps = np.random.uniform(-1, 1, num_layers)
        # 简单的阈值处理，确保反射系数够强
        amps[np.abs(amps) < 0.2] = 0.5 
        r[indices] = amps

        # 扩展到 3D (Z, X, Y) -> 初始是水平层
        r_3d = np.tile(r[:, np.newaxis, np.newaxis], (1, self.n2, self.n3))
        return r_3d

    def apply_folding(self, data, k_max=3):
        """添加褶皱构造"""
        z, x, y = np.indices(self.shape)
        s = np.zeros(self.shape)
        
        # 1. 高斯褶皱
        num_gaussians = np.random.randint(1, k_max + 1)
        for _ in range(num_gaussians):
            x0 = np.random.randint(0, self.n2)
            y0 = np.random.randint(0, self.n3)
            sigma = np.random.uniform(20, 50)
            amp = np.random.uniform(-15, 15)
            gauss = amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
            scale = 1.0 + 0.5 * z / self.n1 # 随深度略微变化
            s += gauss * scale

        # 2. 平面倾斜 (Planar Shearing)
        e0 = np.random.uniform(-5, 5)
        f = np.random.uniform(-0.1, 0.1)
        g = np.random.uniform(-0.1, 0.1)
        s += (e0 + f * x + g * y)

        # 坐标映射: z' = z + s => 我们在 z 处取值，相当于找原来的 z+s 处的值
        coords = np.array([z + s, x, y])
        
        # 使用 linear 插值 (order=1) 保证平滑
        folded_data = ndimage.map_coordinates(data, coords, order=1, mode='nearest')
        return folded_data

    def add_faults(self, reflectivity, num_faults=3):
        """【修改】添加断层，并修复多断层时的标签错位问题"""
        faulted_r = reflectivity.copy()
        fault_label = np.zeros(self.shape)
        
        z, x, y = np.indices(self.shape)
        
        for i in range(num_faults):
            # --- 1. 定义断层几何 ---
            x0 = np.random.randint(0, self.n2)
            y0 = np.random.randint(0, self.n3)
            z0 = np.random.randint(0, self.n1)
            
            theta = np.deg2rad(np.random.uniform(0, 360)) 
            phi = np.deg2rad(np.random.uniform(60, 85))   
            
            n_vec = np.array([np.cos(phi), np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta)])
            dist = n_vec[0]*(z-z0) + n_vec[1]*(x-x0) + n_vec[2]*(y-y0)
            
            # --- 2. 计算位移场 ---
            max_disp = np.random.uniform(5, 15) # 稍微减小位移，避免过于夸张
            shift_mask = 1.0 / (1.0 + np.exp(-dist)) 
            spatial_decay = np.exp(-((x-x0)**2 + (y-y0)**2)/(100**2)) 
            displacement = max_disp * shift_mask * spatial_decay
            
            # 构建坐标映射 (z - displacement)
            # 意味着当前点 (z,x,y) 的值取自 (z-disp, x, y)
            # 这等效于上面的岩石向下移动了 disp
            coords = np.array([z - displacement, x, y])

            # --- 3. 【关键修复】同步移动之前的标签 ---
            # 如果之前的断层已经被画在 label 上了，现在岩石移动了，label 也要跟着动
            if i > 0:
                # order=0 (最近邻) 防止标签变糊，或者用 order=1 后阈值化
                fault_label = ndimage.map_coordinates(fault_label, coords, order=1, mode='nearest')
                # 重新二值化，防止插值产生小数
                fault_label[fault_label > 0.1] = 1.0  
                fault_label[fault_label <= 0.1] = 0.0

            # --- 4. 移动反射系数体 ---
            faulted_r = ndimage.map_coordinates(faulted_r, coords, order=1, mode='nearest')
            
            # --- 5. 添加当前断层的标签 ---
            dist_to_plane = np.abs(dist)
            # 标注宽度设为 1.5 到 2 个像素
            mask_current = dist_to_plane < 0.6
            fault_label = np.maximum(fault_label, mask_current.astype(float))
            
        return faulted_r, fault_label

    def convolve_wavelet(self, reflectivity, target_freq=None):
        """【修改】支持指定主频"""
        if target_freq is None:
            f_peak = np.random.uniform(25, 45) # 默认范围
        else:
            # 在目标频率附近波动，增加多样性
            f_peak = np.random.normal(target_freq, 5) 
            f_peak = np.clip(f_peak, 15, 60) # 限制合理范围

        wavelet = self.ricker_wavelet(f_peak)
        
        # 沿 Z 轴卷积
        seismic = np.apply_along_axis(lambda m: np.convolve(m, wavelet, mode='same'), 0, reflectivity)
        return seismic

    def add_noise(self, data, noise_std=0.1):
        """【修改】支持指定噪声水平"""
        # 实际数据的噪声通常不是均匀的，这里生成不同强度的噪声
        noise = np.random.normal(0, 1, data.shape)
        
        # 缩放到指定信噪比
        signal_std = np.std(data)
        # 加上一个随机波动，让不同样本的噪声不一样
        current_noise_level = noise_std * np.random.uniform(0.8, 1.2)
        
        noisy_data = data + noise * signal_std * current_noise_level
        return noisy_data

    def generate(self, target_freq=35, noise_level=0.1):
        """执行完整流程，传入控制参数"""
        # 1. 稀疏反射系数
        r = self.generate_reflectivity()
        # 2. 褶皱
        r_folded = self.apply_folding(r)
        # 3. 断层
        r_faulted, label = self.add_faults(r_folded)
        # 4. 卷积 (传入频率)
        seismic = self.convolve_wavelet(r_faulted, target_freq=target_freq)
        # 5. 加噪 (传入噪声等级)
        seismic_noisy = self.add_noise(seismic, noise_std=noise_level)
        
        # 6. 最终归一化 (使用 Z-Score)
        mean_val = np.mean(seismic_noisy)
        std_val = np.std(seismic_noisy)
        seismic_final = (seismic_noisy - mean_val) / (std_val + 1e-8)
        
        return seismic_final, label

def generate_dataset(num_samples, output_dir, dataset_name='train', 
                     target_freq=30, noise_level=0.2): # 【修改】默认加大噪声
    """批量生成"""
    seismic_dir = os.path.join(output_dir, dataset_name, 'seis')
    label_dir = os.path.join(output_dir, dataset_name, 'fault')
    os.makedirs(seismic_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    generator = SeismicGenerator(shape=(128, 128, 128))
    
    print(f"生成 {dataset_name} | 目标主频: ~{target_freq}Hz | 噪声等级: {noise_level}")
    
    for i in tqdm(range(num_samples)):
        # 调用 generate 时传入参数
        seismic, fault_label = generator.generate(target_freq=target_freq, noise_level=noise_level)
        
        seismic_path = os.path.join(seismic_dir, f'{i}.npy') # 简化文件名
        label_path = os.path.join(label_dir, f'{i}.npy')
        
        np.save(seismic_path, seismic.astype(np.float32))
        np.save(label_path, fault_label.astype(np.float32))

if __name__ == "__main__":
    # === 配置区域 ===
    # 你的实际数据大概是多少 Hz？如果是 25Hz，这里就填 25
    REAL_DATA_FREQ = 30  
    
    # 你的实际数据有多脏？0.1 是很干净，0.3 是较脏，0.5 是非常脏
    # 建议生成两批：一批中等噪声(0.2)，一批高噪声(0.4)
    NOISE_LEVEL = 0.3    
    
    OUTPUT_DIR = './synthetic_data_v2'
    
    # 生成训练集 (200个)
    generate_dataset(num_samples=200, output_dir=OUTPUT_DIR, dataset_name='train',
                     target_freq=REAL_DATA_FREQ, noise_level=NOISE_LEVEL)
    
    # 生成测试集 (20个)
    generate_dataset(num_samples=20, output_dir=OUTPUT_DIR, dataset_name='prediction',
                     target_freq=REAL_DATA_FREQ, noise_level=NOISE_LEVEL)
    
    # --- 验证部分 ---
    print("\n检查生成的样本...")
    s = np.load(os.path.join(OUTPUT_DIR, 'train/seis/0.npy'))
    l = np.load(os.path.join(OUTPUT_DIR, 'train/fault/0.npy'))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Seismic (Inline 64)")
    plt.imshow(s[:, 64, :], cmap='gray', aspect='auto')
    plt.subplot(122)
    plt.title("Fault (Inline 64)")
    plt.imshow(l[:, 64, :], cmap='jet', aspect='auto')
    plt.savefig('check_sample.png')
    print("检查图已保存为 check_sample.png")