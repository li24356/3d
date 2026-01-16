import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """两次 3x3x3 Conv + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """下采样：MaxPool + DoubleConv3D"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)

class AttentionGate3D(nn.Module):
    """3D注意力门：g = decoder特征, x = encoder skip 特征"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(F_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # g: gating (decoder), x: skip (encoder)
        # assume g and x have same spatial size (上采样后)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        attn = self.sigmoid(psi)
        return x * attn  # 广播相乘

class UpAttn3D(nn.Module):
    """上采样 + Attention Gate + 拼接 + DoubleConv3D"""
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv1x1 = None

        self.attn = AttentionGate3D(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 1))
        self.conv = DoubleConv3D(out_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder 特征（需上采样）， x2: encoder skip 特征
        if self.bilinear:
            x1 = self.up(x1)
            x1 = self.conv1x1(x1) if self.conv1x1 else x1
        else:
            x1 = self.up(x1)

        # 尺寸对齐（3D）
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        
        if diffZ != 0 or diffY != 0 or diffX != 0:
            padding = [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2,
                diffZ // 2, diffZ - diffZ // 2
            ]
            x1 = F.pad(x1, padding)

        # 注意力门作用在 skip 上
        x2_att = self.attn(x1, x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionUNet3D(nn.Module):
    """3D Attention U-Net（输入:128x128x128）"""
    def __init__(self, in_channels=1, out_channels=1, bilinear=False, base_channels=16):
        super().__init__()
        # Encoder路径
        self.inc = DoubleConv3D(in_channels, base_channels)      # 128x128x128 -> [B,16,128,128,128]
        self.down1 = Down3D(base_channels, base_channels*2)      # -> [B,32,64,64,64]
        self.down2 = Down3D(base_channels*2, base_channels*4)    # -> [B,64,32,32,32]
        self.down3 = Down3D(base_channels*4, base_channels*8)    # -> [B,128,16,16,16]
        
        # 可以添加更多下采样层来处理128x128x128的输入
        self.down4 = Down3D(base_channels*8, base_channels*16)   # -> [B,256,8,8,8]

        # Decoder路径
        self.up1 = UpAttn3D(base_channels*16, base_channels*8, base_channels*8, bilinear=bilinear)  # [B,256,8,8,8] -> [B,128,16,16,16]
        self.up2 = UpAttn3D(base_channels*8, base_channels*4, base_channels*4, bilinear=bilinear)   # [B,128,16,16,16] -> [B,64,32,32,32]
        self.up3 = UpAttn3D(base_channels*4, base_channels*2, base_channels*2, bilinear=bilinear)   # [B,64,32,32,32] -> [B,32,64,64,64]
        self.up4 = UpAttn3D(base_channels*2, base_channels, base_channels, bilinear=bilinear)       # [B,32,64,64,64] -> [B,16,128,128,128]

        self.outc = OutConv3D(base_channels, out_channels)

    def forward(self, x):
        # Encoder
        e1 = self.inc(x)      # [B,16,128,128,128]
        e2 = self.down1(e1)   # [B,32,64,64,64]
        e3 = self.down2(e2)   # [B,64,32,32,32]
        e4 = self.down3(e3)   # [B,128,16,16,16]
        e5 = self.down4(e4)   # [B,256,8,8,8]

        # Decoder with skip connections
        d1 = self.up1(e5, e4)  # [B,128,16,16,16]
        d2 = self.up2(d1, e3)  # [B,64,32,32,32]
        d3 = self.up3(d2, e2)  # [B,32,64,64,64]
        d4 = self.up4(d3, e1)  # [B,16,128,128,128]

        out = self.outc(d4)    # [B,1,128,128,128]
        return out


class LightAttentionUNet3D(nn.Module):
    """轻量版3D Attention U-Net（输入:128x128x128，只有3层下采样）"""
    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        super().__init__()
        self.inc = DoubleConv3D(in_channels, 16)      # 128x128x128 -> [B,16,128,128,128]
        self.down1 = Down3D(16, 32)                   # -> [B,32,64,64,64]
        self.down2 = Down3D(32, 64)                   # -> [B,64,32,32,32]
        self.down3 = Down3D(64, 128)                  # -> [B,128,16,16,16]

        self.up1 = UpAttn3D(128, 64, 64, bilinear=bilinear)   # [B,128,16,16,16] -> [B,64,32,32,32]
        self.up2 = UpAttn3D(64, 32, 32, bilinear=bilinear)    # [B,64,32,32,32] -> [B,32,64,64,64]
        self.up3 = UpAttn3D(32, 16, 16, bilinear=bilinear)    # [B,32,64,64,64] -> [B,16,128,128,128]

        self.outc = OutConv3D(16, out_channels)

    def forward(self, x):
        e1 = self.inc(x)      # [B,16,128,128,128]
        e2 = self.down1(e1)   # [B,32,64,64,64]
        e3 = self.down2(e2)   # [B,64,32,32,32]
        e4 = self.down3(e3)   # [B,128,16,16,16]

        d1 = self.up1(e4, e3)  # [B,64,32,32,32]
        d2 = self.up2(d1, e2)  # [B,32,64,64,64]
        d3 = self.up3(d2, e1)  # [B,16,128,128,128]

        out = self.outc(d3)    # [B,1,128,128,128]
        return out


if __name__ == "__main__":
    # 测试完整版
    x = torch.randn(1, 1, 128, 128, 128)
    print("输入形状:", x.shape)
    
    # 完整版（4层下采样）
    model_full = AttentionUNet3D()
    y_full = model_full(x)
    print("完整版输出形状:", y_full.shape)
    
    # 轻量版（3层下采样）
    model_light = LightAttentionUNet3D()
    y_light = model_light(x)
    print("轻量版输出形状:", y_light.shape)
    
    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"完整版参数量: {count_parameters(model_full):,}")
    print(f"轻量版参数量: {count_parameters(model_light):,}")