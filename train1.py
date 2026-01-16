import os
# 允许重复的 OpenMP 运行时
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from datetime import datetime
import random

# ============================================================================
# 用户自定义模块导入 (请确保这些文件在您的目录下)
# ============================================================================
try:
    from models.unet3d import UNet3D
    from dataloader import VolumeDataset
    from models.AERB3d import AERBUNet3DLight
    from models.AERB3d import AERBUNet3D
    from models.attention_unet3d import LightAttentionUNet3D
    from models.seunet3d import SEUNet3D
except ImportError:
    print("⚠️  警告: 未找到模型定义文件，请确保 models 文件夹及相关 .py 文件存在。")
    # 为了防止代码报错，这里定义简单的占位符，实际运行时请忽略
    UNet3D = AERBUNet3DLight = LightAttentionUNet3D = SEUNet3D = VolumeDataset = None

# ============================================================================
# 设置随机种子
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ============================================================================
# 模型配置
# ============================================================================
MODEL_CONFIG = {
    "model_name": "attn_light",  # 可选: 'unet3d', 'aerb_light', 'attn_light', 'seunet3d', 'aerb3d'
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 16,
    "dropout_prob": 0.3,
    "pretrained_ckpt": None,
    "overfit_patience": 500  # 提前停止的耐心值
}

_MODEL_REGISTRY = {
    "unet3d": UNet3D,
    "aerb_light": AERBUNet3DLight,
    "attn_light": LightAttentionUNet3D,
    "seunet3d": SEUNet3D,
    'aerb3d': AERBUNet3D,
}

# ============================================================================
# 优化的混合损失函数 (Dice + BCE)
# ============================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, eps=1e-6):
        """
        组合 Dice Loss 和 BCE Loss
        """
        super(DiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.eps = eps
        # 使用 BCEWithLogitsLoss (内置 Sigmoid，数值更稳定)
        self.bce_func = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # 1. 计算 BCE Loss
        if targets.dtype != torch.float32:
            targets = targets.float()
        bce_loss_val = self.bce_func(logits, targets)

        # 2. 计算 Dice Loss
        probs = torch.sigmoid(logits)
        # 沿用之前的逻辑：在 (dim=1,2,3,4) 上求和
        num = 2 * (probs * targets).sum(dim=(1, 2, 3, 4))
        den = probs.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
        
        dice_score = (num + self.eps) / (den + self.eps)
        dice_loss_val = 1 - dice_score.mean()

        # 3. 加权组合
        total_loss = (self.weight_dice * dice_loss_val) + (self.weight_bce * bce_loss_val)
        
        return total_loss, dice_loss_val, bce_loss_val


def augment_batch_data(x, y):
    """
    根据论文实现的 3D 地震数据增强
    1. 随机沿深度轴 (Z轴) 翻转
    2. 随机绕深度轴 (Z轴) 旋转 0, 90, 180, 270 度
    
    参数:
        x: 地震数据 Tensor, 形状 (B, C, D, H, W) 或 (B, D, H, W)
        y: 标签数据 Tensor, 形状同 x
    """
    # 确保是 Tensor
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(np.asarray(y))

    # 获取维度索引
    # 假设标准 5D 输入: (Batch, Channel, Depth, Height, Width) -> (B, C, Z, Y, X)
    # Depth 是 dim=2, H-W 平面是 dim=[3, 4]
    if x.ndim == 5:
        z_axis = 2
        plane_axes = [3, 4]
    elif x.ndim == 4: # (Batch, Depth, Height, Width)
        z_axis = 1
        plane_axes = [2, 3]
    else:
        return x, y # 维度不对，跳过增强

    # --- 1. 垂直翻转 (Vertical Flip / Z-flip) ---
    if random.random() > 0.5:
        x = torch.flip(x, dims=[z_axis])
        y = torch.flip(y, dims=[z_axis])

    # --- 2. 绕 Z 轴旋转 (Rotation 90/180/270) ---
    # 这相当于在 H-W (Inline-Crossline) 平面上旋转
    k = random.randint(0, 3) # 生成 0, 1, 2, 3
    if k > 0:
        x = torch.rot90(x, k, dims=plane_axes)
        y = torch.rot90(y, k, dims=plane_axes)

    return x, y



# ============================================================================
# 辅助函数：优化器、调度器、归一化、IoU
# ============================================================================
def get_optimizer(model, lr=1e-4, optimizer_type='adamw', weight_decay=1e-3):
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return optimizer

def get_adaptive_scheduler(optimizer, warmup_epochs=None, total_epochs=400, scheduler_type='auto'):
    if warmup_epochs is None:
        if total_epochs <= 30: warmup_epochs = max(2, total_epochs // 10)
        elif total_epochs <= 100: warmup_epochs = max(5, total_epochs // 10)
        elif total_epochs <= 200: warmup_epochs = max(10, total_epochs // 15)
        else: warmup_epochs = max(15, total_epochs // 20)
    
    if scheduler_type == 'auto':
        # [修改] 建议对于 400 epoch 这种长训练，直接使用单周期余弦 'cosine'
        # 这样学习率会平滑下降，不会在后期突然反弹，有利于模型收敛稳定
        scheduler_type = 'cosine' 
    
    print(f"调度配置: {total_epochs} epochs -> warmup={warmup_epochs}, scheduler={scheduler_type}")
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    
    if scheduler_type == 'cosine_warm':
        # 如果你非要用热重启，建议把 T_0 设小一点，比如 50
        T_0 = 50 
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)
    elif scheduler_type == 'cosine':
        # 这是最稳的策略
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    else:
        step_size = max(20, total_epochs // 5)
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    return scheduler, None


def _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0)):
    """鲁棒归一化 (MAD)"""
    if isinstance(x, torch.Tensor): x_np = x.cpu().numpy()
    else: x_np = np.asarray(x)
    original_shape = x_np.shape
    
    if x_np.ndim == 5: x_reshaped = x_np.reshape(original_shape[0] * original_shape[1], *original_shape[2:])
    elif x_np.ndim == 4: x_reshaped = x_np
    else: raise ValueError('Unsupported ndim')
    
    x_norm = np.zeros_like(x_reshaped, dtype=np.float32)
    for i in range(x_reshaped.shape[0]):
        sample = x_reshaped[i]
        center = np.median(sample)
        if use_mad:
            mad = np.median(np.abs(sample - center))
            scale = mad * 1.4826 if mad > 1e-6 else 1.0
        else:
            scale = np.std(sample)
            if scale < 1e-6: scale = 1.0
        sample_norm = (sample - center) / scale
        if clip_range: sample_norm = np.clip(sample_norm, clip_range[0], clip_range[1])
        x_norm[i] = sample_norm
        
    if x_np.ndim == 5: x_norm = x_norm.reshape(original_shape)
    return torch.from_numpy(x_norm).float()

def _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0)):
    """传统归一化 (Mean/Std)"""
    if isinstance(x, torch.Tensor): x = x.cpu().numpy()
    x = np.asarray(x)
    spatial_axes = (2, 3, 4) if x.ndim == 5 else (1, 2, 3)
    mean = x.mean(axis=spatial_axes, keepdims=True)
    std = x.std(axis=spatial_axes, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std
    if clip_range: x_norm = np.clip(x_norm, clip_range[0], clip_range[1])
    return torch.from_numpy(x_norm).float()

def _batch_iou(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold)
    targets_bool = targets.bool()
    intersection = (preds & targets_bool).sum(dim=(1, 2, 3, 4)).float()
    union = (preds | targets_bool).sum(dim=(1, 2, 3, 4)).float()
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

# ============================================================================
# 过拟合检测器
# ============================================================================
class OverfittingDetector:
    def __init__(self, patience=15, min_delta=1e-4, gap_threshold=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.gap_threshold = gap_threshold
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_val_gaps = []

    def update(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_val_gaps.append(abs(train_loss - val_loss))
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            improved = False
        return improved, self.check_overfitting()

    def check_overfitting(self):
        signals = []
        if len(self.val_losses) >= 5:
            recent = self.val_losses[-5:]
            if np.polyfit(range(5), recent, 1)[0] > 0.01: signals.append("验证损失连续上升")
        if self.train_val_gaps[-1] > self.gap_threshold: signals.append("训练-验证差距过大")
        return signals

    def should_stop(self, epoch):
        return (self.counter >= self.patience), self.check_overfitting()
    
    def get_summary(self):
        return f"Best: {self.best_val_loss:.6f} (Epoch {self.best_epoch}) | Patience: {self.counter}/{self.patience}"

# ============================================================================
# 绘图工具
# ============================================================================
def plot_extended_training_curves(train_losses, val_losses, train_ious, val_ious, 
                                  learning_rates=None, save_path=None):
    """绘制扩展的训练曲线"""
    if learning_rates is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs_range, train_losses, 'b-', label='Train', alpha=0.7)
    ax.plot(epochs_range, val_losses, 'r-', label='Val', alpha=0.7)
    ax.set_title('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # IoU
    ax = axes[0, 1]
    ax.plot(epochs_range, train_ious, 'g-', label='Train', alpha=0.7)
    ax.plot(epochs_range, val_ious, 'm-', label='Val', alpha=0.7)
    ax.set_title('IoU')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Gap
    ax = axes[1, 0]
    if len(train_losses) > 0:
        gaps = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        ax.plot(epochs_range, gaps, 'c-', label='Train-Val Gap')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Loss Gap')
        ax.legend(); ax.grid(True, alpha=0.3)

    # IoU Gap
    ax = axes[1, 1]
    if len(train_ious) > 0:
        iou_gaps = [abs(t - v) for t, v in zip(train_ious, val_ious)]
        ax.plot(epochs_range, iou_gaps, 'y-', label='IoU Gap')
        ax.set_title('IoU Gap')
        ax.legend(); ax.grid(True, alpha=0.3)
        
    # LR
    if learning_rates is not None:
        ax = axes[0, 2] if len(axes.shape) == 2 else axes[2]
        ax.plot(epochs_range, learning_rates, color='purple')
        ax.set_yscale('log')
        ax.set_title('Learning Rate')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=100)
    plt.close()

# ============================================================================
# 训练和验证函数
# ============================================================================
def train_epoch(model, loader, opt, criterion, device, scaler=None, accum_steps=1, max_grad_norm=1.0, use_robust_norm=True):
    model.train()
    stats = {'loss': 0.0, 'bce': 0.0, 'dice': 0.0, 'iou': 0.0}
    opt.zero_grad()
    
    for step, (x, y) in enumerate(tqdm(loader, desc='train', leave=False), start=1):
        x, y = augment_batch_data(x, y)
        if use_robust_norm:
            x = _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0)).to(device)
        else:
            x = _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0)).to(device)
        y = y.float().to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss, dice_val, bce_val = criterion(logits, y)
                batch_iou = _batch_iou(logits, y)
            scaler.scale(loss / accum_steps).backward()
        else:
            logits = model(x)
            loss, dice_val, bce_val = criterion(logits, y)
            batch_iou = _batch_iou(logits, y)
            (loss / accum_steps).backward()

        bs = x.size(0)
        stats['loss'] += loss.item() * bs
        stats['bce'] += bce_val.item() * bs
        stats['dice'] += dice_val.item() * bs
        stats['iou'] += batch_iou.item() * bs

        if step % accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
            opt.zero_grad()
            
    total = len(loader.dataset)
    return {k: v / total for k, v in stats.items()}

def validate(model, loader, criterion, device, use_robust_norm=True):
    model.eval()
    stats = {'loss': 0.0, 'bce': 0.0, 'dice': 0.0, 'iou': 0.0}
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc='val', leave=False):
            if use_robust_norm:
                x = _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0)).to(device)
            else:
                x = _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0)).to(device)
            y = y.float().to(device)
            
            logits = model(x)
            loss, dice_val, bce_val = criterion(logits, y)
            batch_iou = _batch_iou(logits, y)
            
            bs = x.size(0)
            stats['loss'] += loss.item() * bs
            stats['bce'] += bce_val.item() * bs
            stats['dice'] += dice_val.item() * bs
            stats['iou'] += batch_iou.item() * bs
            
    total = len(loader.dataset)
    return {k: v / total for k, v in stats.items()}

def build_model_from_config(cfg):
    name = cfg.get("model_name", "unet3d")
    cls = _MODEL_REGISTRY.get(name)
    if cls is None: raise ValueError(f"Unknown model: {name}")
    
    kwargs = {k: v for k, v in cfg.items() if k in cls.__init__.__code__.co_varnames}
    model = cls(**kwargs)
    
    ckpt = cfg.get("pretrained_ckpt")
    if ckpt and os.path.exists(ckpt):
        try:
            sd = torch.load(ckpt, map_location="cpu")
            if "model_state" in sd: sd = sd["model_state"]
            model.load_state_dict(sd, strict=False)
            print("已加载预训练权重。")
        except Exception as e:
            print(f"权重加载失败: {e}")
    return model

# ============================================================================
# 主函数
# ============================================================================
def main():
    # ========== 1. 核心训练配置 ==========
    WEIGHT_BCE = 0.5   
    WEIGHT_DICE = 0.5  
    
    USE_ROBUST_NORM = True 
    NORM_CLIP = (-4.0, 4.0) if USE_ROBUST_NORM else (-3.0, 3.0)

    EPOCHS = 400
    BATCH_SIZE = 8
    LR = 1e-4
    WORKERS = 4
    
    # [新增] .dat 文件专用配置 (必须与你的数据一致)
    DAT_CONFIG = {
        'dat_dtype': 'float32',       # 数据类型
        'dat_shape': (128, 128, 128), # 数据尺寸 (Z, H, W)
        'dat_order': 'C'              # 字节序
    }
    
    # 路径配置
    root = Path('.')
    train_data = root / 'train' / 'seis'
    train_label = root / 'train' / 'fault'
    val_data = root / 'prediction' / 'seis'
    val_label = root / 'prediction' / 'fault'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # ========== 2. 数据准备 (已修复) ==========
    if not train_data.exists():
        print(f"⚠️ 错误: 训练数据未找到 {train_data}")
        return 

    # [修复] 传入 **DAT_CONFIG 解包参数
    ds_train = VolumeDataset(str(train_data), str(train_label), **DAT_CONFIG)
    
    if val_data.exists():
        ds_val = VolumeDataset(str(val_data), str(val_label), **DAT_CONFIG)
    else:
        print("⚠️ 提示: 未找到验证集，自动划分 10% 训练集用于验证。")
        train_len = int(0.9 * len(ds_train))
        ds_train, ds_val = torch.utils.data.random_split(ds_train, [train_len, len(ds_train)-train_len])
    
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=min(2, WORKERS))
    
    # ========== 3. 模型与优化器 ==========
    model = build_model_from_config(MODEL_CONFIG).to(device)
    opt = get_optimizer(model, lr=LR, optimizer_type='adamw')
    
    criterion = DiceBCELoss(weight_dice=WEIGHT_DICE, weight_bce=WEIGHT_BCE).to(device)
    
    scheduler, plateau_scheduler = get_adaptive_scheduler(opt, total_epochs=EPOCHS)

    detector = OverfittingDetector(patience=MODEL_CONFIG.get("overfit_patience", 500))
    
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda') else None
    
    # ========== 4. 保存目录准备 ==========
    checkpoints_root = Path('checkpoints2')
    model_tag = MODEL_CONFIG["model_name"]
    base_ch = MODEL_CONFIG.get('base_channels')
    if base_ch and f"_c{base_ch}" not in model_tag: model_tag = f"{model_tag}_c{base_ch}"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_dir = checkpoints_root / model_tag / timestamp
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    latest_dir = checkpoints_root / model_tag / 'latest'
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_save_dir / 'training_config.txt', 'w') as f:
        f.write(f"Model: {model_tag}\nWeights: BCE={WEIGHT_BCE}, Dice={WEIGHT_DICE}\n")
        f.write(f"Norm: Robust={USE_ROBUST_NORM}, Clip={NORM_CLIP}\n")
        f.write(f"LR: {LR}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}\n")
        f.write(f"Dat Config: {DAT_CONFIG}\n") # 记录数据配置

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'lr': []}
    best_val_iou = 0.0

    print(f"\n开始训练 | Loss Weights: BCE={WEIGHT_BCE}/Dice={WEIGHT_DICE} | Norm: {NORM_CLIP}\n")
    
    # ========== 5. 训练循环 ==========
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        
        t_metrics = train_epoch(model, loader_train, opt, criterion, device, scaler=scaler, use_robust_norm=USE_ROBUST_NORM)
        v_metrics = validate(model, loader_val, criterion, device, use_robust_norm=USE_ROBUST_NORM)
        
        history['train_loss'].append(t_metrics['loss'])
        history['val_loss'].append(v_metrics['loss'])
        history['train_iou'].append(t_metrics['iou'])
        history['val_iou'].append(v_metrics['iou'])
        history['lr'].append(opt.param_groups[0]['lr'])
        
        print(f" Train | Loss: {t_metrics['loss']:.4f} (B:{t_metrics['bce']:.3f}, D:{t_metrics['dice']:.3f}) | IoU: {t_metrics['iou']:.4f}")
        print(f" Val   | Loss: {v_metrics['loss']:.4f} (B:{v_metrics['bce']:.3f}, D:{v_metrics['dice']:.3f}) | IoU: {v_metrics['iou']:.4f}")
        
        improved, signals = detector.update(epoch, t_metrics['loss'], v_metrics['loss'])
        print(f" {detector.get_summary()}")
        
        scheduler.step()
        
        last_state = {
            'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
            'train_loss': t_metrics['loss'], 'val_loss': v_metrics['loss'],
            'train_iou': t_metrics['iou'], 'val_iou': v_metrics['iou'],
            'history': history, 'config': MODEL_CONFIG
        }
        
        torch.save(last_state, model_save_dir / 'model_last.pth')
        torch.save(last_state, latest_dir / 'model_last.pth')
        
        if improved:
            torch.save(last_state, model_save_dir / 'model_best_loss.pth')
            torch.save(last_state, latest_dir / 'model_best_loss.pth')
            print(f" ✅ Saved Best Loss Model")

        if v_metrics['iou'] > best_val_iou:
            best_val_iou = v_metrics['iou']
            torch.save(last_state, model_save_dir / 'model_best_iou.pth')
            torch.save(last_state, latest_dir / 'model_best_iou.pth')
            print(f" ✅ Saved Best IoU Model ({best_val_iou:.4f})")
            
        if epoch % 5 == 0 or epoch == EPOCHS:
            plot_path = model_save_dir / f'training_curve_epoch_{epoch}.png'
            plot_extended_training_curves(
                history['train_loss'], history['val_loss'], 
                history['train_iou'], history['val_iou'], 
                history['lr'], plot_path
            )
            plot_extended_training_curves(
                history['train_loss'], history['val_loss'], 
                history['train_iou'], history['val_iou'], 
                history['lr'], latest_dir / 'training_curve_latest.png'
            )

        should_stop, _ = detector.should_stop(epoch)
        if should_stop:
            print("🛑 Early stopping triggered.")
            break

    print("训练结束。")
if __name__ == '__main__':
    main()