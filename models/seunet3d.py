import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation Block adapted for 3D data.
    Reference: Hu et al., "Squeeze-and-Excitation Networks"
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Scale
        return x * y.expand_as(x)

class DoubleConvSE(nn.Module):
    """
    Module with two 3D convolutions followed by an SE Block.
    Structure: Conv3d -> BN -> ReLU -> Conv3d -> BN -> ReLU -> SEBlock
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock3D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

class SEUNet3D(nn.Module):
    """
    SEU-Net architecture for 3D Seismic Fault Identification.
    Input shape: (Batch, 1, 128, 128, 128)
    Output shape: (Batch, 1, 128, 128, 128)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(SEUNet3D, self).__init__()
        
        # --- Encoder (Downsampling Path) ---
        # Level 1: 32 channels
        self.enc1 = DoubleConvSE(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        # Level 2: 64 channels
        self.enc2 = DoubleConvSE(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        # Level 3: 128 channels
        self.enc3 = DoubleConvSE(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # --- Bridge (Bottom) ---
        # Bottom: 256 channels
        self.bridge = DoubleConvSE(128, 256)
        
        # --- Decoder (Upsampling Path) ---
        # Level 3 Up: 256 -> 128
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConvSE(256, 128) # Concat(128+128) -> 256 input inside block? 
        # Note: Standard U-Net conv block takes concatenated channels.
        # Here: In_channels = 128 (from up) + 128 (from skip) = 256. Out = 128.
        self.dec3_block = DoubleConvSE(256, 128)
        
        # Level 2 Up: 128 -> 64
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        # In_channels = 64 (from up) + 64 (from skip) = 128. Out = 64.
        self.dec2_block = DoubleConvSE(128, 64)
        
        # Level 1 Up: 64 -> 32
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        # In_channels = 32 (from up) + 32 (from skip) = 64. Out = 32.
        self.dec1_block = DoubleConvSE(64, 32)
        
        # --- Output Layer ---
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # -> (B, 32, D, H, W)
        p1 = self.pool1(e1)     # -> (B, 32, D/2, H/2, W/2)
        
        e2 = self.enc2(p1)      # -> (B, 64, D/2, H/2, W/2)
        p2 = self.pool2(e2)     # -> (B, 64, D/4, H/4, W/4)
        
        e3 = self.enc3(p2)      # -> (B, 128, D/4, H/4, W/4)
        p3 = self.pool3(e3)     # -> (B, 128, D/8, H/8, W/8)
        
        # Bridge
        b = self.bridge(p3)     # -> (B, 256, D/8, H/8, W/8)
        
        # Decoder
        d3 = self.up3(b)        # -> (B, 128, D/4, H/4, W/4)
        # Skip connection: Concatenate along channel axis (dim=1)
        d3 = torch.cat((e3, d3), dim=1) # -> (B, 256, ...)
        d3 = self.dec3_block(d3) # -> (B, 128, ...)
        
        d2 = self.up2(d3)       # -> (B, 64, D/2, H/2, W/2)
        d2 = torch.cat((e2, d2), dim=1) # -> (B, 128, ...)
        d2 = self.dec2_block(d2) # -> (B, 64, ...)
        
        d1 = self.up1(d2)       # -> (B, 32, D, H, W)
        d1 = torch.cat((e1, d1), dim=1) # -> (B, 64, ...)
        d1 = self.dec1_block(d1) # -> (B, 32, ...)
        
        # Output
        out = self.final_conv(d1)
        return self.sigmoid(out)

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEUNet3D(in_channels=1, out_channels=1).to(device)
    
    # 创建虚拟 3D 数据 (Batch_Size, Channels, Depth, Height, Width)
    # 输入尺寸: 128 x 128 x 128
    input_data = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 前向传播
    output = model(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
