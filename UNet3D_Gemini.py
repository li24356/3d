import os
import time
import random
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from scipy.ndimage import gaussian_filter

# =========================================================
# 0. 全局配置 & 随机种子
# =========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================================================
# 1. 3D 基础模块 (ResBlock3D, SE3D, ASPP3D)
# =========================================================

class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation Block"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 4)),  # 防止通道过小
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 4), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Global Average Pooling in 3D: (B, C, D, H, W) -> (B, C)
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ResBlock3D(nn.Module):
    """3D Residual Block"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 使用 reflection padding 可以减少 3D 边界效应
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SEBlock3D(out_ch)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # 嵌入 SE 模块
        return F.relu(x + res)


class ASPP3D(nn.Module):
    """
    3D 空洞空间金字塔池化
    针对三维数据的感受野扩展
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1x1x1 卷积
        self.conv1 = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)

        # 3x3x3 卷积, rate=2 (3D dilation)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        # 3x3x3 卷积, rate=4
        self.conv3 = nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Sequential(
            nn.Conv3d(out_ch * 3, in_ch, 1, bias=False),  # 融合后投射回原维度
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3)  # 增加 Dropout 防止过拟合
        )

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))

        res = torch.cat([x1, x2, x3], dim=1)
        return self.project(res)


# =========================================================
# 2. 核心网络：FaultNet3D (U-Net 架构)
# =========================================================

class FaultNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        # 为了节省显存，base_channels 设为 16 或 32
        # 如果你有 24G+ 显存，可以设为 32
        c = base_channels

        # Encoder (Downsampling)
        self.enc1 = ResBlock3D(in_channels, c)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResBlock3D(c, c * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResBlock3D(c * 2, c * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck (ASPP Enhanced)
        self.bot = ResBlock3D(c * 4, c * 8)
        self.aspp = ASPP3D(c * 8, c * 2)  # 内部降维处理

        # Decoder (Upsampling)
        # 3D 上采样使用 trilinear 插值
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ResBlock3D(c * 8 + c * 4, c * 4)  # Concat后通道数增加

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ResBlock3D(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ResBlock3D(c * 2 + c, c)

        self.outc = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        e1 = self.enc1(x)  # -> c
        e2 = self.enc2(self.pool1(e1))  # -> 2c
        e3 = self.enc3(self.pool2(e2))  # -> 4c

        b = self.bot(self.pool3(e3))  # -> 8c
        b = self.aspp(b)  # -> 8c (feature refined)

        # Decoder with Skip Connections
        d3 = self.up3(b)
        # 简单的 padding 处理防止尺寸不匹配
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.outc(d1)