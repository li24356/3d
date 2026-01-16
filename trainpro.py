import os
# 允许重复的 OpenMP运行时（不推荐作为长期解决方案，只是临时变通）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from datetime import datetime
import random

from models.unet3d import UNet3D
from dataloader import VolumeDataset
from models.AERB3d import AERBUNet3DLight
from models.attention_unet3d import LightAttentionUNet3D
from models.seunet3d import SEUNet3D
from models.AERB3d import AERBUNet3D
from models.attention_unet3d import AttentionUNet3D
from models.CBAM import CBAM_UNet3D


# ======= TF32 设置（RTX 3090 强烈推荐）=======
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ============================================

class DiceFocalLoss(nn.Module):
    """
    【全能型损失函数】
    Dice Loss (关注全局/形状/重叠度) + Focal Loss (关注局部/难分像素/边缘)
    """
    def __init__(self, weight_dice=1.0, weight_focal=1.0, gamma=2.0, alpha=0.25):
        super(DiceFocalLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.gamma = gamma       # 聚焦参数：越大越关注难分样本（如断层边缘）
        self.alpha = alpha       # 类别平衡参数：用于调整正负样本的权重基础值
        self.eps = 1e-6

    def forward(self, logits, targets):
        """
        输入:
            logits: [B, 1, D, H, W] (模型输出，未经过 Sigmoid)
            targets: [B, 1, D, H, W] (标签，0或1)
        """
        # 0. 确保数据类型一致
        if targets.dtype != logits.dtype:
            targets = targets.type_as(logits)
            
        # --------------------------------------------
        # 1. 计算 Dice Loss (关注整体重叠度)
        # --------------------------------------------
        probs = torch.sigmoid(logits)
        
        # 展平计算，防止 Batch 间干扰
        batch_size = logits.shape[0]
        probs_flat = probs.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice_score = (2. * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1 - dice_score.mean()

        # --------------------------------------------
        # 2. 计算 Focal Loss (关注难分样本)
        # --------------------------------------------
        # 计算每个像素的 BCE Loss (不进行归约 reduction='none')
        # bce_loss = - [y * log(p) + (1-y) * log(1-p)]
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # [优化] 明确正负样本权重逻辑
        # alpha_t: target=1时为alpha(0.75), target=0时为1-alpha(0.25)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return self.weight_dice * dice_loss + self.weight_focal * focal_loss.mean() 

# ============================================================================
# 设置随机种子
# ============================================================================
def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置随机种子
set_seed(42)

# ============================================================================
# 可编辑模型配置（在此处修改以训练不同模型）
# ============================================================================
MODEL_CONFIG = {
    "model_name": "attention_unet3d",  # 可选: 'unet3d', 'aerb_light', 'attn_light', 'seunet3d', 'aerb3d', 'attention_unet3d'
    "in_channels": 1,
    "out_channels": 1,
    "pretrained_ckpt": None,
}

# 简单模型注册表与构造器
_MODEL_REGISTRY = {
    "unet3d": UNet3D,
    "aerb_light": AERBUNet3DLight,
    "attn_light": LightAttentionUNet3D,
    "seunet3d": SEUNet3D,
    'aerb3d': AERBUNet3D,
    'attention_unet3d': AttentionUNet3D,
    'cbam_unet3d': CBAM_UNet3D,
}

def augment_batch_data(x, y):
    """
    【回退版】仅保留水平/垂直翻转。
    去掉了 Transpose，因为实际测试证明地震数据的方向性差异过大，不适合强制交换。
    """
    # 1. 随机水平翻转 (X轴镜像)
    if random.random() > 0.5:
        x = torch.flip(x, dims=[4]) 
        y = torch.flip(y, dims=[4])
        
    # 2. 随机垂直翻转 (Y轴镜像)
    if random.random() > 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])

    
    # 3. [新增] 随机 90度 旋转 (在 X-Y 平面)
    # 地震切片通常在水平方向上没有绝对的方向性
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k, dims=[3, 4])
        y = torch.rot90(y, k, dims=[3, 4])    

    return x, y

# ============================================================================
# 优化器函数（改进版）
# ============================================================================
def get_optimizer(model, lr=1e-4, optimizer_type='adamw', weight_decay=1e-4):
    """
    获取适合3D分割网络的优化器
    
    Args:
        model: 要优化的模型
        lr: 初始学习率
        optimizer_type: 'adamw', 'adam', 'sgd'
        weight_decay: L2正则化强度
    """
    if optimizer_type.lower() == 'adamw':
        # AdamW: 更好的权重衰减处理，通常性能更好
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),      # 动量参数
            eps=1e-8,                # 数值稳定性
            weight_decay=weight_decay,  # L2正则化，防止过拟合
            amsgrad=False            # 通常不需要AMSGrad
        )
    elif optimizer_type.lower() == 'adam':
        # 改进的Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay  # 添加权重衰减
        )
    elif optimizer_type.lower() == 'sgd':
        # SGD with momentum (对某些任务有效)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,           # 动量
            weight_decay=weight_decay,  # 权重衰减
            nesterov=True           # Nesterov动量
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def build_model_from_config(cfg):
    """
    根据 MODEL_CONFIG 构造模型实例并（可选）加载预训练权重。
    直接编辑上方 MODEL_CONFIG 即可切换模型与参数。
    """
    name = cfg.get("model_name", "unet3d")
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model_name '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")

    # 使用模型默认参数实例化
    kwargs = {}
    if "in_channels" in cfg:
        kwargs["in_channels"] = cfg["in_channels"]
    if "out_channels" in cfg:
        kwargs["out_channels"] = cfg["out_channels"]

    # 实例化
    model = cls(**{k: v for k, v in kwargs.items() if k in cls.__init__.__code__.co_varnames})

    # 加载预训练权重（若提供路径）
    ckpt = cfg.get("pretrained_ckpt")
    if ckpt:
        ckpt_path = Path(ckpt)
        if ckpt_path.exists():
            sd = torch.load(str(ckpt_path), map_location="cpu")
            # 支持常见包装字段
            if isinstance(sd, dict) and "model_state" in sd:
                sd = sd["model_state"]
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # 去掉 module. 前缀（DataParallel）
            new_sd = {}
            for k, v in sd.items():
                nk = k[len("module."):] if k.startswith("module.") else k
                new_sd[nk] = v
            try:
                model.load_state_dict(new_sd, strict=False)
                print(f"Loaded pretrained weights from {ckpt_path} (strict=False)")
            except Exception as e:
                print(f"Warning: failed to fully load pretrained weights: {e}")
        else:
            print(f"Warning: pretrained_ckpt path does not exist: {ckpt_path}")

    return model

# ============================================================================
# 损失函数
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3, 4))
        den = probs.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
        dice = (num + self.eps) / (den + self.eps)
        return 1 - dice.mean()

# ============================================================================
# 过拟合检测器
# ============================================================================
class OverfittingDetector:
    """
    检测过拟合的类
    检测标准：
    1. 验证损失连续上升（相对于最低点）
    2. 训练损失和验证损失差距过大
    3. Dice系数等指标出现分歧
    """
    def __init__(self, patience=10, min_delta=1e-4, gap_threshold=0.5, history_window=5):
        """
        Args:
            patience: 容忍验证损失不下降的epoch数
            min_delta: 验证损失的最小改善阈值
            gap_threshold: 训练损失和验证损失差距的阈值
            history_window: 用于计算趋势的窗口大小
        """
        self.patience = patience
        self.min_delta = min_delta
        self.gap_threshold = gap_threshold
        self.history_window = history_window
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        
        # 记录历史
        self.train_losses = []
        self.val_losses = []
        self.train_val_gaps = []
        
    def update(self, epoch, train_loss, val_loss):
        """更新检测器状态"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        gap = abs(train_loss - val_loss)
        self.train_val_gaps.append(gap)
        
        # 检查是否是最佳验证损失
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            improved = False
            
        return improved, self.check_overfitting(epoch)
    
    def check_overfitting(self, epoch):
        """检查是否过拟合"""
        overfitting_signals = []
        
        # 1. 检查验证损失是否连续上升
        if len(self.val_losses) >= self.history_window:
            recent_val = self.val_losses[-self.history_window:]
            val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
            if val_trend > 0.01:  # 正斜率，验证损失上升
                overfitting_signals.append(f"验证损失连续上升 (趋势: {val_trend:.4f})")
        
        # 2. 检查训练损失和验证损失的差距
        if len(self.train_val_gaps) > 0:
            recent_gap = self.train_val_gaps[-1]
            if recent_gap > self.gap_threshold:
                overfitting_signals.append(f"训练-验证损失差距过大 ({recent_gap:.4f} > {self.gap_threshold})")
        
        # 3. 检查训练损失持续下降但验证损失不下降
        if len(self.train_losses) >= self.history_window:
            recent_train = self.train_losses[-self.history_window:]
            train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
            if train_trend < -0.01 and self.counter >= self.patience // 2:
                overfitting_signals.append("训练损失持续下降但验证损失停滞")
        
        return overfitting_signals
    
    def get_summary(self):
        """获取检测器状态摘要"""
        if len(self.train_losses) == 0:
            return "无训练历史"
        
        summary = [
            f"最佳验证损失: {self.best_val_loss:.6f} (epoch {self.best_epoch})",
            f"当前验证损失: {self.val_losses[-1]:.6f}",
            f"当前训练损失: {self.train_losses[-1]:.6f}",
            f"训练-验证差距: {self.train_val_gaps[-1]:.6f}",
            f"连续未改善epoch数: {self.counter}/{self.patience}"
        ]
        
        # 添加趋势信息
        if len(self.val_losses) >= 3:
            recent_val = self.val_losses[-3:]
            val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
            trend_symbol = '↑' if val_trend > 0.001 else ('↓' if val_trend < -0.001 else '→')
            summary.append(f"验证损失趋势: {trend_symbol} ({val_trend:.6f})")
        
        return "\n".join(summary)
    
    def should_stop(self, epoch):
        """判断是否应该停止训练"""
        overfitting_signals = self.check_overfitting(epoch)
        # 主要停止条件：验证损失连续patience次未改善
        should_stop = (self.counter >= self.patience)
        return should_stop, overfitting_signals

# ============================================================================
# 可视化工具
# ============================================================================
def plot_training_curves(train_losses, val_losses, learning_rates=None, save_path=None):
    """绘制训练曲线"""
    if learning_rates is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax = axes[0] if learning_rates is None else axes[0]
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 差距曲线
    ax = axes[1] if learning_rates is None else axes[1]
    if len(train_losses) > 0 and len(val_losses) > 0:
        gaps = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        ax.plot(epochs, gaps, 'g-', label='Train-Val Gap', alpha=0.7, linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Warning Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gap')
        ax.set_title('Train-Validation Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 学习率曲线
    if learning_rates is not None:
        ax = axes[2]
        # 使用显式 color/linestyle，避免像 'purple-' 这样的非法 format string
        ax.plot(epochs, learning_rates, color='purple', linestyle='-', label='Learning Rate', alpha=0.7, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 归一化函数（鲁棒版本）
# ============================================================================
def _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0), device=None):
    """
    GPU加速版鲁棒Z-score标准化：使用中位数和MAD（中位数绝对偏差）
    直接在GPU上执行，避免CPU-GPU数据往返
    
    Args:
        x: 输入张量，形状 (B, C, Z, Y, X) 或 (B, Z, Y, X)
        use_mad: 是否使用MAD（中位数绝对偏差）代替标准差
        clip_range: 截断范围
        device: 计算设备（如果为None则使用x的设备）
    """
    # 确保x是torch张量且在GPU上
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x)).float()
    
    if device is None:
        device = x.device
    
    x = x.to(device)
    
    if x.ndim == 5:  # (B, C, Z, Y, X)
        spatial_axes = (2, 3, 4)
        n_samples = x.shape[0] * x.shape[1]
        x_reshaped = x.reshape(n_samples, -1)
    elif x.ndim == 4:  # (B, Z, Y, X)
        spatial_axes = (1, 2, 3)
        n_samples = x.shape[0]
        x_reshaped = x.reshape(n_samples, -1)
    else:
        raise ValueError('Unsupported input ndim for normalization: %d' % x.ndim)
    
    # GPU上计算中位数
    center = torch.median(x_reshaped, dim=1, keepdim=True)[0]
    
    if use_mad:
        # 计算MAD（中位数绝对偏差）
        abs_dev = torch.abs(x_reshaped - center)
        mad = torch.median(abs_dev, dim=1, keepdim=True)[0]
        # MAD -> std: σ ≈ 1.4826 * MAD
        scale = torch.where(mad > 1e-6, mad * 1.4826, torch.ones_like(mad))
    else:
        # 使用标准差
        scale = torch.std(x_reshaped, dim=1, keepdim=True)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    
    # 归一化
    x_norm = (x_reshaped - center) / scale
    
    # 截断异常值
    if clip_range:
        x_norm = torch.clamp(x_norm, clip_range[0], clip_range[1])
    
    # 恢复原始形状
    if x.ndim == 5:
        x_norm = x_norm.reshape(x.shape)
    elif x.ndim == 4:
        x_norm = x_norm.reshape(x.shape)
    
    return x_norm.float()


def _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0), device=None):
    """
    GPU加速版传统Z-score标准化
    直接在GPU上执行，避免CPU-GPU数据往返
    
    Args:
        x: 输入张量
        clip_range: 截断范围
        device: 计算设备
    """
    # 确保x是torch张量且在GPU上
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x)).float()
    
    if device is None:
        device = x.device
    
    x = x.to(device)
    
    if x.ndim == 5:  # (B, C, Z, Y, X)
        n_samples = x.shape[0] * x.shape[1]
        x_reshaped = x.reshape(n_samples, -1)
    elif x.ndim == 4:  # (B, Z, Y, X)
        n_samples = x.shape[0]
        x_reshaped = x.reshape(n_samples, -1)
    else:
        raise ValueError('Unsupported input ndim for normalization: %d' % x.ndim)
    
    # GPU上计算均值和标准差
    mean = torch.mean(x_reshaped, dim=1, keepdim=True)
    std = torch.std(x_reshaped, dim=1, keepdim=True)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    
    # 归一化
    x_norm = (x_reshaped - mean) / std
    
    # 截断异常值
    if clip_range:
        x_norm = torch.clamp(x_norm, clip_range[0], clip_range[1])
    
    # 恢复原始形状
    if x.ndim == 5:
        x_norm = x_norm.reshape(x.shape)
    elif x.ndim == 4:
        x_norm = x_norm.reshape(x.shape)
    
    return x_norm.float()


def _batch_iou(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold)
    targets_bool = targets.bool()
    intersection = (preds & targets_bool).sum(dim=(1, 2, 3, 4)).float()
    union = (preds | targets_bool).sum(dim=(1, 2, 3, 4)).float()
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

# ============================================================================
# 归一化效果分析函数
# ============================================================================
def analyze_normalization_effect(loader, use_robust_norm=True, num_batches=5):
    """
    分析归一化效果
    """
    print("\n" + "="*60)
    print("归一化效果分析")
    print("="*60)
    
    before_stats = {'mean': [], 'std': [], 'median': [], 'mad': [], 'min': [], 'max': []}
    after_stats = {'mean': [], 'std': [], 'median': [], 'mad': [], 'min': [], 'max': []}
    
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
            
        # 原始统计
        x_np = x.numpy()
        before_stats['mean'].append(np.mean(x_np))
        before_stats['std'].append(np.std(x_np))
        before_stats['median'].append(np.median(x_np))
        before_stats['mad'].append(np.median(np.abs(x_np - np.median(x_np))))
        before_stats['min'].append(np.min(x_np))
        before_stats['max'].append(np.max(x_np))
        
        # 归一化后统计
        if use_robust_norm:
            x_norm = _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0)).numpy()
        else:
            x_norm = _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0)).numpy()
        
        after_stats['mean'].append(np.mean(x_norm))
        after_stats['std'].append(np.std(x_norm))
        after_stats['median'].append(np.median(x_norm))
        after_stats['mad'].append(np.median(np.abs(x_norm - np.median(x_norm))))
        after_stats['min'].append(np.min(x_norm))
        after_stats['max'].append(np.max(x_norm))
    
    print("\n归一化前后对比:")
    print("="*60)
    print("             归一化前                归一化后")
    print("="*60)
    
    for stat_name in ['mean', 'std', 'median', 'mad', 'min', 'max']:
        before_avg = np.mean(before_stats[stat_name])
        before_std = np.std(before_stats[stat_name])
        after_avg = np.mean(after_stats[stat_name])
        after_std = np.std(after_stats[stat_name])
        
        print(f"{stat_name:6s}: {before_avg:8.4f} ± {before_std:.4f}  ->  {after_avg:8.4f} ± {after_std:.4f}")
    
    print("="*60)
    print(f"归一化方法: {'鲁棒Z-score(MAD)' if use_robust_norm else '传统Z-score'}")
    print(f"截断范围: {(-4.0, 4.0) if use_robust_norm else (-3.0, 3.0)}")
    
    return before_stats, after_stats

# ============================================================================
# 训练和验证函数（改进版，带梯度裁剪）
# ============================================================================
def train_epoch(model, loader, opt, criterion,
                device, scaler=None, accum_steps=1, max_grad_norm=1.0,
                use_robust_norm=True,scheduler=None):
    """
    训练一个epoch，包含梯度裁剪
    
    Args:
        max_grad_norm: 梯度裁剪的最大范数
        use_robust_norm: 是否使用鲁棒归一化
    """
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_iou = 0.0
    opt.zero_grad()
    
    for step, (x, y) in enumerate(tqdm(loader, desc='train', leave=False), start=1):
        # ===== 直接转到GPU，移除增强步骤 =====
        x = x.to(device)
        y = y.float().to(device)
        
        x, y = augment_batch_data(x, y)
        
        # ===== GPU上执行归一化 =====
        if use_robust_norm:
            x = _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0), device=device)
        else:
            x = _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0), device=device)

        # 混合精度前向
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)  # 直接调用 DiceFocalLoss
                batch_iou = _batch_iou(logits, y)
            scaler.scale(loss / accum_steps).backward()
        else:
            logits = model(x)
            loss = criterion(logits, y)      # 直接调用 DiceFocalLoss
            batch_iou = _batch_iou(logits, y)
            (loss / accum_steps).backward()

        # 记录指标（与验证保持一致）
        with torch.no_grad():
            # 纯 BCE 观察分类误差
            bce = F.binary_cross_entropy_with_logits(logits, y)
            # 纯 Dice 观察重叠度
            probs = torch.sigmoid(logits)
            intersection = (probs * y).sum()
            union = probs.sum() + y.sum()
            dice = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        running_bce += bce.item() * batch_size
        running_dice += dice.item() * batch_size
        running_iou += batch_iou.item() * batch_size


        # 梯度累积：每 accum_steps 步更新一次
        if (step % accum_steps) == 0:
            if scaler is not None:
                # 混合精度训练时的梯度裁剪
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                # 普通训练时的梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
            opt.zero_grad()

            if scheduler is not None:
                scheduler.step()
    
    # 若最后不足 accum_steps，也要执行一次 step（安全处理）
    if len(loader) % accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        opt.zero_grad()

        if scheduler is not None:
            scheduler.step()
    
    total = len(loader.dataset)
    return {
        'total_loss': running_loss / total,
        'bce_loss': running_bce / total,
        'dice_loss': running_dice / total,
        'iou': running_iou / total
    }

def validate(model, loader, criterion, device, use_robust_norm=True):
    """
    验证函数（已修正：与训练Loss保持一致）
    """
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    # 定义辅助计算函数（仅用于日志显示，不影响反向传播或Loss计算）
    # 这里的 BCE 用于观察纯粹的分类误差
    def calc_bce(logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)
        
    # 这里的 Dice 用于观察纯粹的重叠度
    def calc_dice(logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice

    with torch.no_grad():
        for x, y in tqdm(loader, desc='val', leave=False):
            # 1. 归一化（保持与训练一致）
            if use_robust_norm:
                # 注意：这里也需要加上 device=device 以防万一，或者依赖 .to(device)
                x = _robust_normalize_tensor_batch(x, use_mad=True, clip_range=(-4.0, 4.0), device=device)
            else:
                x = _traditional_normalize_tensor_batch(x, clip_range=(-3.0, 3.0), device=device)
            
            x = x.to(device)
            y = y.float().to(device)
            
            logits = model(x)
            
            # ====================================================
            # 2. 核心修改：使用与训练完全相同的 Loss 函数
            # ====================================================
            # 之前是手动加权 0.3*BCE + 0.7*Dice，这是错误的。
            # 现在直接调用传入的 DiceFocalLoss
            loss = criterion(logits, y) 
            
            # ====================================================
            # 3. 辅助指标计算 (仅用于显示/日志)
            # ====================================================
            # 为了让日志里的 bce_loss 和 dice_loss 有数据，我们单独算一下
            # 注意：这里的 loss_dice 是纯 Dice，而 criterion 里面可能包含 Focal
            val_bce = calc_bce(logits, y)
            val_dice = calc_dice(logits, y)
            
            batch_iou = _batch_iou(logits, y)
            
            # 累加
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            running_bce += val_bce.item() * batch_size
            running_dice += val_dice.item() * batch_size
            running_iou += batch_iou.item() * batch_size
    
    total = len(loader.dataset)
    return {
        'total_loss': running_loss / total, # 这是最重要的指标，用于 Early Stopping
        'bce_loss': running_bce / total,    # 仅作参考
        'dice_loss': running_dice / total,  # 仅作参考
        'iou': running_iou / total          # 重要的评价指标
    }

# ============================================================================
# 主函数
# ============================================================================
def main():
    # ========== 训练配置 ==========
    root = Path('.')
    epochs = 200               # 总训练epoch数（自适应调度器会自动调整）
    batch_size = 8              # 批大小
    lr = 1e-4                    # 初始学习率
    workers = 8               # 数据加载线程数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========== 归一化配置 ==========
    use_robust_normalization = True  # 是否使用鲁棒归一化（推荐True）
    norm_clip_range = (-4.0, 4.0) if use_robust_normalization else (-3.0, 3.0)
    
    # ========== 优化器配置 ==========
    optimizer_type = 'adamw'        # 可选: 'adamw', 'adam', 'sgd'
    weight_decay = 1e-4             # L2正则化强度
    max_grad_norm = 1.0             # 梯度裁剪阈值
    
    # ========== 调度器配置 ==========
    scheduler_type = 'one_cycle'         # 'auto'（自动选择）、'cosine_warm'、'cosine'、'plateau'、'step'
    warmup_epochs = None            # None表示自动计算
    
    # ========== 过拟合检测参数 ==========
    overfit_patience = 500          # 容忍验证损失不下降的epoch数
    overfit_min_delta = 1e-4        # 最小改善阈值
    overfit_gap_threshold = 0.5     # 训练-验证损失差距阈值
    
    # ========== 其他训练配置 ==========
    use_amp = True                  # 混合精度训练
    accum_steps = 1                 # 梯度累积步数
    
    # 数据格式配置
    dat_dtype = 'float32'
    dat_shape = (128, 128, 128)
    dat_order = 'C'
    
    # ========== 数据路径 ==========
    train_data = root / 'train' / 'seis'
    train_label = root / 'train' / 'fault'
    val_data = root / 'prediction' / 'seis'
    val_label = root / 'prediction' / 'fault'
    
    print('=' * 70)
    print('训练配置')
    print('=' * 70)
    print(f'总Epochs: {epochs}')
    print(f'归一化方法: {"鲁棒Z-score(MAD)" if use_robust_normalization else "传统Z-score"}')
    print(f'归一化截断范围: {norm_clip_range}')
    print(f'损失权重: BCE=0.3, Dice=0.7')
    print(f'优化器: {optimizer_type}, LR: {lr}, Weight Decay: {weight_decay}')
    print(f'调度器类型: {scheduler_type}')
    print(f'梯度裁剪: {max_grad_norm}')
    print(f'设备: {device}')
    print(f'模型: {MODEL_CONFIG["model_name"]}')
    print(f'批大小: {batch_size}, 工作线程: {workers}')
    print('=' * 70)
    
    # ========== 数据路径检查 ==========
    print('检查数据路径...')
    if not train_data.exists():
        raise FileNotFoundError(f"训练数据路径不存在: {train_data}")
    if not train_label.exists():
        raise FileNotFoundError(f"训练标签路径不存在: {train_label}")
    if not val_data.exists():
        print(f"⚠️  警告: 验证数据路径不存在: {val_data}")
        print(f"⚠️  注意: 将无法进行验证，训练将继续但无法检测过拟合")
        # 这里可以选择创建空验证集或调整逻辑
    if not val_label.exists():
        print(f"⚠️  警告: 验证标签路径不存在: {val_label}")
    
    # ========== 数据加载 ==========
    print('加载数据...')
    ds_train = VolumeDataset(str(train_data), str(train_label), 
                            dat_dtype=dat_dtype, dat_shape=dat_shape, dat_order=dat_order)
    
    # 检查验证集路径是否存在
    if val_data.exists() and val_label.exists():
        ds_val = VolumeDataset(str(val_data), str(val_label), 
                              dat_dtype=dat_dtype, dat_shape=dat_shape, dat_order=dat_order)
    else:
        # 如果没有验证集，使用训练集的一部分作为验证（不推荐但可以继续训练）
        print("⚠️  使用训练集的前10%作为验证集（仅用于演示，实际使用时请提供验证集）")
        from torch.utils.data import random_split
        train_size = int(0.9 * len(ds_train))
        val_size = len(ds_train) - train_size
        ds_train, ds_val = random_split(ds_train, [train_size, val_size], 
                                       generator=torch.Generator().manual_seed(42))
    
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, 
                             num_workers=workers, pin_memory=True, 
                             persistent_workers=True if workers > 0 else False)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False, 
                           num_workers=min(2, workers), pin_memory=False)
    
    print(f'训练集大小: {len(ds_train)}')
    print(f'验证集大小: {len(ds_val)}')
    
    # ========== 归一化效果分析 ==========
    analyze_normalization_effect(
        loader_train, 
        use_robust_norm=use_robust_normalization,
        num_batches=3
    )
    
    # ========== 模型初始化 ==========
    device = torch.device(device)
    model = build_model_from_config(MODEL_CONFIG)
    model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型总参数量: {total_params:,}')
    print(f'可训练参数量: {trainable_params:,}')
    
    # ========== 优化器初始化 ==========
    opt = get_optimizer(
        model, 
        lr=lr, 
        optimizer_type=optimizer_type,
        weight_decay=weight_decay
    )
    
    # ========== 损失函数 ==========
    # alpha=0.25 (根据原论文，配合gamma=2使用)
    # gamma=2.0 (标准聚焦参数)
    # weight_dice=1.0, weight_focal=1.0 (两者同等重要，或者 dice=0.8, focal=1.2 强调边缘)
    criterion = DiceFocalLoss(weight_dice=1.0, weight_focal=1.0, gamma=2.0, alpha=0.75)
    criterion.to(device)
    
    # ========== 自适应调度器初始化 ==========
    steps_per_epoch = len(loader_train) 

    # 【优化】计算实际的 step 次数
    # 如果使用了梯度累积，优化器更新的次数是 总batch数 / accum_steps
    # 我们需要确保 OneCycle 的总步数与优化器实际 step 的次数一致
    effective_steps_per_epoch = steps_per_epoch // accum_steps
    if steps_per_epoch % accum_steps != 0:
         effective_steps_per_epoch += 1 # 加上最后一个不完整的 batch
        
    total_steps = steps_per_epoch * epochs

    print(f"One-Cycle 配置: Max LR=1e-3, Total Steps={total_steps}")

    # 2. 定义调度器
    # max_lr 通常设为初始 lr 的 10 倍左右。
    # 3D U-Net 比较脆弱，如果你原来用 1e-4，这里 max_lr 建议给 1e-3 或 5e-4
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=1e-3,           # 最大学习率峰值
        total_steps=total_steps,
        pct_start=0.3,         # 前30%的时间用来升温(Warmup)
        div_factor=25.0,       # 初始学习率 = max_lr / 25
        final_div_factor=1e4,  # 最终学习率 = 初始学习率 / 10000
        anneal_strategy='cos'  # 使用余弦形状
    )

    # 注意：OneCycle 不需要 plateau_scheduler，把这行删掉或设为 None
    plateau_scheduler = None
    
    # ========== 过拟合检测器初始化 ==========
    overfit_detector = OverfittingDetector(
        patience=overfit_patience,
        min_delta=overfit_min_delta,
        gap_threshold=overfit_gap_threshold
    )
    
    # ========== 保存目录设置 ==========
    checkpoints_root = Path('checkpoints3')

    # 修改这里：直接使用 model_name 作为 model_tag
    model_tag = MODEL_CONFIG["model_name"]  # 简化逻辑
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_save_dir = checkpoints_root / model_tag / timestamp
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 方便快速访问最近一次运行：checks/.../<model_tag>/latest/
    latest_dir = checkpoints_root / model_tag / 'latest'
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练配置
    config_save_path = model_save_dir / 'training_config.txt'
    with open(config_save_path, 'w') as f:
        f.write(f"训练配置\n")
        f.write(f"========\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"归一化方法: {'鲁棒Z-score(MAD)' if use_robust_normalization else '传统Z-score'}\n")
        f.write(f"归一化截断范围: {norm_clip_range}\n")
        f.write(f"损失权重: BCE=0.3, Dice=0.7\n")
        f.write(f"优化器: {optimizer_type}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"权重衰减: {weight_decay}\n")
        f.write(f"调度器类型: {scheduler_type}\n")
        f.write(f"批大小: {batch_size}\n")
        f.write(f"工作线程: {workers}\n")
        f.write(f"AMP混合精度: {'启用' if use_amp else '禁用'}\n")
        f.write(f"模型: {MODEL_CONFIG['model_name']}\n")
        f.write(f"模型参数量: {total_params:,}\n")
    
    # ========== AMP scaler ==========
    if use_amp and device.type != 'cuda':
        print("⚠️  警告: AMP仅在CUDA设备上可用，已禁用AMP")
        use_amp = False
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
    
    # ========== 记录变量 ==========
    best_val_loss = float('inf')
    best_val_iou = 0.0
    train_losses_history = []
    val_losses_history = []
    train_ious_history = []
    val_ious_history = []
    learning_rates_history = []
    
    # ========== 训练循环 ==========
    print('\n开始训练...')
    for epoch in range(1, epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{epochs}')
        print(f'{"="*60}')
        
        # 训练
        train_metrics = train_epoch(
            model, loader_train, opt, criterion,
            device, scaler=scaler, accum_steps=accum_steps, 
            max_grad_norm=max_grad_norm,
            use_robust_norm=use_robust_normalization,scheduler=scheduler
        )
        train_loss = train_metrics['total_loss']
        
        # 验证
        val_metrics = validate(
            model, loader_val, criterion, device,
            use_robust_norm=use_robust_normalization
        )
        val_loss = val_metrics['total_loss']
        
        # 记录历史
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        train_ious_history.append(train_metrics['iou'])
        val_ious_history.append(val_metrics['iou'])
        learning_rates_history.append(opt.param_groups[0]['lr'])
        
        # 打印结果
        print(f"  Train Loss: {train_loss:.6f} (BCE: {train_metrics['bce_loss']:.6f}, "
              f"Dice: {train_metrics['dice_loss']:.6f}, IoU: {train_metrics['iou']:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} (BCE: {val_metrics['bce_loss']:.6f}, "
              f"Dice: {val_metrics['dice_loss']:.6f}, IoU: {val_metrics['iou']:.6f})")
        print(f"  当前学习率: {opt.param_groups[0]['lr']:.2e}")
        
        # ========== 更新过拟合检测器 ==========
        improved, overfitting_signals = overfit_detector.update(epoch, train_loss, val_loss)
        
        # 显示过拟合检测器状态
        print(f"\n  {overfit_detector.get_summary()}")
        
        # 检查是否过拟合
        should_stop, stop_reasons = overfit_detector.should_stop(epoch)
        if overfitting_signals:
            print(f"  ⚠️  过拟合信号:")
            for signal in overfitting_signals:
                print(f"     - {signal}")
        
        
        # ========== 保存模型 ==========
        # 准备调度器状态
        scheduler_state = None
        plateau_scheduler_state = None
        
        if scheduler_type != 'plateau':
            scheduler_state = scheduler.state_dict()
        if plateau_scheduler is not None:
            plateau_scheduler_state = plateau_scheduler.state_dict()
        
        # 保存最新模型
        last_path = model_save_dir / 'model_last.pth'
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'scheduler_state': scheduler_state,
            'plateau_scheduler_state': plateau_scheduler_state,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'train_history': train_losses_history,
            'val_history': val_losses_history,
            'train_ious': train_ious_history,
            'val_ious': val_ious_history,
            'lr_history': learning_rates_history,
            'model_config': MODEL_CONFIG,
            'normalization_method': 'robust' if use_robust_normalization else 'traditional'
        }, str(last_path))
        
        # 也写一份到 latest/
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'model_config': MODEL_CONFIG,
            'normalization_method': 'robust' if use_robust_normalization else 'traditional'
        }, str(latest_dir / 'model_last.pth'))
        
        # 保存最佳模型（基于验证损失）
        save_best_loss = False
        save_best_iou = False
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_loss = True
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_best_iou = True
        
        if save_best_loss:
            best_path = model_save_dir / 'model_best_loss.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'train_history': train_losses_history,
                'val_history': val_losses_history,
                'train_ious': train_ious_history,
                'val_ious': val_ious_history,
                'lr_history': learning_rates_history,
                'model_config': MODEL_CONFIG,
                'normalization_method': 'robust' if use_robust_normalization else 'traditional'
            }, str(best_path))
            
            # 更新 latest 中的 best_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'model_config': MODEL_CONFIG,
                'normalization_method': 'robust' if use_robust_normalization else 'traditional'
            }, str(latest_dir / 'model_best_loss.pth'))
            
            print(f'  ✅ 保存最佳损失模型 (val_loss: {val_loss:.6f}, val_iou: {val_metrics["iou"]:.6f})')
        
        if save_best_iou:
            best_iou_path = model_save_dir / 'model_best_iou.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'train_history': train_losses_history,
                'val_history': val_losses_history,
                'train_ious': train_ious_history,
                'val_ious': val_ious_history,
                'lr_history': learning_rates_history,
                'model_config': MODEL_CONFIG,
                'normalization_method': 'robust' if use_robust_normalization else 'traditional'
            }, str(best_iou_path))
            
            # 更新 latest 中的 best_iou
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'model_config': MODEL_CONFIG,
                'normalization_method': 'robust' if use_robust_normalization else 'traditional'
            }, str(latest_dir / 'model_best_iou.pth'))
            
            print(f'  ✅ 保存最佳IoU模型 (val_iou: {val_metrics["iou"]:.6f}, val_loss: {val_loss:.6f})')
        
        # ========== 定期保存训练曲线 ==========
        if epoch % 10 == 0 or epoch == epochs or should_stop:
            # 扩展的绘图函数
            def plot_extended_training_curves(train_losses, val_losses, train_ious, val_ious, 
                                            learning_rates=None, save_path=None):
                """绘制扩展的训练曲线，包括损失和IoU"""
                if learning_rates is None:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                else:
                    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                
                epochs_range = range(1, len(train_losses) + 1)
                
                # 损失曲线
                ax = axes[0, 0]
                ax.plot(epochs_range, train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=2)
                ax.plot(epochs_range, val_losses, 'r-', label='Val Loss', alpha=0.7, linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # IoU曲线
                ax = axes[0, 1]
                ax.plot(epochs_range, train_ious, 'g-', label='Train IoU', alpha=0.7, linewidth=2)
                ax.plot(epochs_range, val_ious, 'm-', label='Val IoU', alpha=0.7, linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('IoU')
                ax.set_title('Training and Validation IoU')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 损失差距曲线
                ax = axes[1, 0]
                if len(train_losses) > 0 and len(val_losses) > 0:
                    gaps = [abs(t - v) for t, v in zip(train_losses, val_losses)]
                    ax.plot(epochs_range, gaps, 'c-', label='Train-Val Gap', alpha=0.7, linewidth=2)
                    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Warning Threshold')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Gap')
                    ax.set_title('Train-Validation Gap')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # IoU差距曲线
                ax = axes[1, 1]
                if len(train_ious) > 0 and len(val_ious) > 0:
                    iou_gaps = [abs(t - v) for t, v in zip(train_ious, val_ious)]
                    ax.plot(epochs_range, iou_gaps, 'y-', label='IoU Gap', alpha=0.7, linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('IoU Gap')
                    ax.set_title('Train-Validation IoU Gap')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 学习率曲线
                if learning_rates is not None:
                    ax = axes[0, 2] if len(axes.shape) == 2 else axes[2]
                    ax.plot(epochs_range, learning_rates, color='purple', linestyle='-', 
                           label='Learning Rate', alpha=0.7, linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Learning Rate')
                    ax.set_title('Learning Rate Schedule')
                    ax.set_yscale('log')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            plot_path = model_save_dir / f'training_curve_epoch_{epoch}.png'
            plot_extended_training_curves(
                train_losses_history, val_losses_history, 
                train_ious_history, val_ious_history,
                learning_rates_history, plot_path
            )
            # 另存一份到 latest
            plot_extended_training_curves(
                train_losses_history, val_losses_history,
                train_ious_history, val_ious_history,
                learning_rates_history, 
                latest_dir / 'training_curve_latest.png'
            )
            print(f"  📊 训练曲线已保存至: {plot_path}")
        
        # ========== 清理缓存 ==========
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # ========== 检查是否应该停止训练 ==========
        if should_stop:
            print(f"\n{'!'*70}")
            print("检测到过拟合，停止训练!")
            print(f"停止原因:")
            for reason in stop_reasons:
                print(f"  - {reason}")
            print(f"最佳模型在 epoch {overfit_detector.best_epoch}, val_loss: {overfit_detector.best_val_loss:.6f}")
            print(f"{'!'*70}")
            
            # 保存过拟合检测结果
            overfit_info_path = model_save_dir / 'overfitting_report.txt'
            with open(overfit_info_path, 'w') as f:
                f.write("过拟合检测报告\n")
                f.write("="*50 + "\n")
                f.write(f"训练停止于 epoch {epoch}\n")
                f.write(f"总训练epoch数: {epochs}\n")
                f.write(f"实际训练epoch数: {epoch}\n")
                f.write(f"最佳 epoch: {overfit_detector.best_epoch}\n")
                f.write(f"最佳验证损失: {overfit_detector.best_val_loss:.6f}\n")
                f.write(f"最佳验证IoU: {best_val_iou:.6f}\n")
                f.write(f"停止原因:\n")
                for reason in stop_reasons:
                    f.write(f"  - {reason}\n")
                f.write("\n训练历史:\n")
                f.write(f"  最终训练损失: {train_loss:.6f}\n")
                f.write(f"  最终验证损失: {val_loss:.6f}\n")
                f.write(f"  最终训练IoU: {train_metrics['iou']:.6f}\n")
                f.write(f"  最终验证IoU: {val_metrics['iou']:.6f}\n")
                f.write(f"  训练-验证损失差距: {abs(train_loss - val_loss):.6f}\n")
                f.write(f"  训练-验证IoU差距: {abs(train_metrics['iou'] - val_metrics['iou']):.6f}\n")
                f.write(f"  最终学习率: {opt.param_groups[0]['lr']:.2e}\n")
                f.write(f"  模型: {MODEL_CONFIG['model_name']}\n")
                f.write(f"  模型参数量: {total_params:,}\n")
                f.write(f"  归一化方法: {'鲁棒Z-score(MAD)' if use_robust_normalization else '传统Z-score'}\n")
            
            print(f"过拟合报告已保存至: {overfit_info_path}")
            
            # 训练提前停止后，保存最终训练曲线
            final_plot_path = model_save_dir / 'training_curve_final.png'
            plot_extended_training_curves(
                train_losses_history, val_losses_history,
                train_ious_history, val_ious_history,
                learning_rates_history, final_plot_path
            )
            break
    
    # ========== 训练完成（未提前停止） ==========
    if not should_stop:
        print(f"\n{'='*70}")
        print("训练完成，未检测到明显过拟合")
        print(f"最佳模型在 epoch {overfit_detector.best_epoch}, val_loss: {overfit_detector.best_val_loss:.6f}")
        print(f"最佳验证IoU: {best_val_iou:.6f}")
        print(f"{'='*70}")
        
        # 保存训练报告
        report_path = model_save_dir / 'training_report.txt'
        with open(report_path, 'w') as f:
            f.write("训练报告\n")
            f.write("="*50 + "\n")
            f.write(f"总Epochs: {epochs}\n")
            f.write(f"完成Epochs: {epochs}\n")
            f.write(f"最佳验证损失: {best_val_loss:.6f}\n")
            f.write(f"最佳验证IoU: {best_val_iou:.6f}\n")
            f.write(f"最终训练损失: {train_loss:.6f}\n")
            f.write(f"最终验证损失: {val_loss:.6f}\n")
            f.write(f"最终训练IoU: {train_metrics['iou']:.6f}\n")
            f.write(f"最终验证IoU: {val_metrics['iou']:.6f}\n")
            f.write(f"训练-验证损失差距: {abs(train_loss - val_loss):.6f}\n")
            f.write(f"训练-验证IoU差距: {abs(train_metrics['iou'] - val_metrics['iou']):.6f}\n")
            f.write(f"优化器: {optimizer_type}\n")
            f.write(f"调度器: {scheduler_type}\n")
            f.write(f"初始学习率: {lr}\n")
            f.write(f"最终学习率: {opt.param_groups[0]['lr']:.2e}\n")
            f.write(f"权重衰减: {weight_decay}\n")
            f.write(f"梯度裁剪: {max_grad_norm}\n")
            f.write(f"模型: {MODEL_CONFIG['model_name']}\n")
            f.write(f"模型参数量: {total_params:,}\n")
            f.write(f"AMP混合精度: {'启用' if use_amp else '禁用'}\n")
            f.write(f"批大小: {batch_size}\n")
            f.write(f"梯度累积步数: {accum_steps}\n")
            f.write(f"归一化方法: {'鲁棒Z-score(MAD)' if use_robust_normalization else '传统Z-score'}\n")
            f.write(f"归一化截断范围: {norm_clip_range}\n")
            f.write(f"损失权重: BCE=0.3, Dice=0.7\n")
            f.write(f"工作线程: {workers}\n")
        
        print(f"训练报告已保存至: {report_path}")

if __name__ == '__main__':
    main()