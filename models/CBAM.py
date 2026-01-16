import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        # 3D 池化: 输出大小为 1x1x1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 共享 MLP (使用 Conv3d kernel=1 实现)
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 3D 卷积: 处理 [Avg; Max] 两个通道
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度 (dim=1) 上做平均和最大操作
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class DoubleConv3D(nn.Module):
    """(Conv3d => BN3d => ReLU) * 2 + CBAM3D"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM3D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x

class Down3D(nn.Module):
    """Downscaling with maxpool3d then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # 3D 上采样: trilinear (三线性插值) 或 ConvTranspose3d
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理可能的尺寸不匹配 (Padding)
        # 输入维度是 [Batch, Channel, Depth, Height, Width]
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        # F.pad 的顺序是从最后一个维度向前: (Left, Right, Top, Bottom, Front, Back)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CBAM_UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, trilinear=False):
        super(CBAM_UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        # Encoder (4层下采样)
        self.inc = DoubleConv3D(n_channels, 32) # 第一层通道数设为32，防止显存爆炸
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(256, 512 // factor)

        # Decoder (4层上采样)
        self.up1 = Up3D(512, 256 // factor, trilinear)
        self.up2 = Up3D(256, 128 // factor, trilinear)
        self.up3 = Up3D(128, 64 // factor, trilinear)
        self.up4 = Up3D(64, 32, trilinear)
        
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --- 验证代码 ---
if __name__ == "__main__":
    # 配置
    input_channels = 1   # 例如: 单模态 MRI/CT
    num_classes = 2      # 例如: 肿瘤 vs 背景
    model = CBAM_UNet3D(n_channels=input_channels, n_classes=num_classes, trilinear=True)
    
    # 打印参数量大致估算
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 创建 128x128x128 的 3D 输入 (Batch Size = 1)
    # Shape: [Batch, Channel, Depth, Height, Width]
    x = torch.randn(1, input_channels, 128, 128, 128)
    
    try:
        y = model(x)
        print(f"\nInput shape:  {x.shape}")
        print(f"Output shape: {y.shape}")
        print("Build 3D CBAM-UNet Successfully!")
    except Exception as e:
        print(f"Error: {e}")
