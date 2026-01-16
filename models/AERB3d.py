import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    """3D轻量级通道注意力机制"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享权重的多层感知机
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化并行
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        # 特征融合
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att

class SpatialAttention3D(nn.Module):
    """3D轻量级空间注意力机制"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 特征拼接
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(spatial_att)
        return x * self.sigmoid(spatial_att)

class LightResidualBlock3D(nn.Module):
    """3D带注意力机制的轻量级残差块"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # 主卷积路径
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        # 注意力模块
        if use_attention:
            self.channel_att = ChannelAttention3D(out_channels)
            self.spatial_att = SpatialAttention3D()
        
        # self.dropout = nn.Dropout3d(p=dropout_prob)  # 3D Dropout
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.dropout(out)  # 只在训练时启用
        out = self.bn2(self.conv2(out))
        
        # 应用注意力机制
        if self.use_attention:
            out = self.channel_att(out)
            out = self.spatial_att(out)
        
        out += identity
        return F.relu(out)

class AERBUNet3D(nn.Module):
    """3D增强版UNet，所有层都使用注意力机制，输入128x128x128，只保留主输出"""
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=0.1, base_channels=16):
        super().__init__()
        
        # 编码器（所有层都使用注意力）
        self.enc1 = LightResidualBlock3D(in_channels, base_channels, dropout_prob, use_attention=True)
        self.enc2 = LightResidualBlock3D(base_channels, base_channels*2, dropout_prob, use_attention=True)
        self.enc3 = LightResidualBlock3D(base_channels*2, base_channels*4, dropout_prob, use_attention=True)
        self.enc4 = LightResidualBlock3D(base_channels*4, base_channels*8, dropout_prob, use_attention=True)
        self.pool = nn.MaxPool3d(2)
        
        # 瓶颈层（带注意力）
        self.bottleneck = nn.Sequential(
            LightResidualBlock3D(base_channels*8, base_channels*16, dropout_prob, use_attention=True),
            LightResidualBlock3D(base_channels*16, base_channels*16, dropout_prob, use_attention=True),
        )
        
        # 解码器（所有层都使用注意力）
        self.up1 = nn.ConvTranspose3d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec1 = LightResidualBlock3D(base_channels*16, base_channels*8, dropout_prob, use_attention=True)
        
        self.up2 = nn.ConvTranspose3d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec2 = LightResidualBlock3D(base_channels*8, base_channels*4, dropout_prob, use_attention=True)
        
        self.up3 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = LightResidualBlock3D(base_channels*4, base_channels*2, dropout_prob, use_attention=True)
        
        self.up4 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.dec4 = LightResidualBlock3D(base_channels*2, base_channels, dropout_prob, use_attention=True)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)                     # [B, 16, 128, 128, 128]
        e2 = self.enc2(self.pool(e1))         # [B, 32, 64, 64, 64]
        e3 = self.enc3(self.pool(e2))         # [B, 64, 32, 32, 32]
        e4 = self.enc4(self.pool(e3))         # [B, 128, 16, 16, 16]
        
        # 瓶颈层
        bn = self.bottleneck(self.pool(e4))   # [B, 256, 8, 8, 8]
        
        # 解码器
        d1 = self.up1(bn)                     # [B, 128, 16, 16, 16]
        d1 = torch.cat([e4, d1], dim=1)       # [B, 256, 16, 16, 16]
        d1 = self.dec1(d1)                    # [B, 128, 16, 16, 16]
        
        d2 = self.up2(d1)                     # [B, 64, 32, 32, 32]
        d2 = torch.cat([e3, d2], dim=1)       # [B, 128, 32, 32, 32]
        d2 = self.dec2(d2)                    # [B, 64, 32, 32, 32]
        
        d3 = self.up3(d2)                     # [B, 32, 64, 64, 64]
        d3 = torch.cat([e2, d3], dim=1)       # [B, 64, 64, 64, 64]
        d3 = self.dec3(d3)                    # [B, 32, 64, 64, 64]
        
        d4 = self.up4(d3)                     # [B, 16, 128, 128, 128]
        d4 = torch.cat([e1, d4], dim=1)       # [B, 32, 128, 128, 128]
        d4 = self.dec4(d4)                    # [B, 16, 128, 128, 128]
        
        # 主输出
        return self.out(d4)

class AERBUNet3DLight(nn.Module):
    """轻量版3D增强UNet（3层下采样），输入128x128x128，只保留主输出"""
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=0.1):
        super().__init__()
        
        # 编码器
        self.enc1 = LightResidualBlock3D(in_channels, 16, dropout_prob, use_attention=True)
        self.enc2 = LightResidualBlock3D(16, 32, dropout_prob, use_attention=True)
        self.enc3 = LightResidualBlock3D(32, 64, dropout_prob, use_attention=True)
        self.pool = nn.MaxPool3d(2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            LightResidualBlock3D(64, 128, dropout_prob, use_attention=True),
            LightResidualBlock3D(128, 128, dropout_prob, use_attention=True),
        )
        
        # 解码器
        self.up1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec1 = LightResidualBlock3D(128, 64, dropout_prob, use_attention=True)
        
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = LightResidualBlock3D(64, 32, dropout_prob, use_attention=True)
        
        self.up3 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec3 = LightResidualBlock3D(32, 16, dropout_prob, use_attention=True)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)                     # [B, 16, 128, 128, 128]
        e2 = self.enc2(self.pool(e1))         # [B, 32, 64, 64, 64]
        e3 = self.enc3(self.pool(e2))         # [B, 64, 32, 32, 32]
        
        # 瓶颈层
        bn = self.bottleneck(self.pool(e3))   # [B, 128, 16, 16, 16]
        
        # 解码器
        d1 = self.up1(bn)                     # [B, 64, 32, 32, 32]
        d1 = torch.cat([e3, d1], dim=1)       # [B, 128, 32, 32, 32]
        d1 = self.dec1(d1)                    # [B, 64, 32, 32, 32]
        
        d2 = self.up2(d1)                     # [B, 32, 64, 64, 64]
        d2 = torch.cat([e2, d2], dim=1)       # [B, 64, 64, 64, 64]
        d2 = self.dec2(d2)                    # [B, 32, 64, 64, 64]
        
        d3 = self.up3(d2)                     # [B, 16, 128, 128, 128]
        d3 = torch.cat([e1, d3], dim=1)       # [B, 32, 128, 128, 128]
        d3 = self.dec3(d3)                    # [B, 16, 128, 128, 128]
        
        # 主输出
        return self.out(d3)

# 测试函数
def test_models():
    # 创建测试数据
    x = torch.randn(1, 1, 128, 128, 128)
    print(f"输入形状: {x.shape}")
    
    # 测试完整版
    print("\n=== 完整版 AERBUNet3D ===")
    model_full = AERBUNet3D(base_channels=16)
    y_full = model_full(x)
    print(f"输出形状: {y_full.shape}")
    
    # 测试轻量版
    print("\n=== 轻量版 AERBUNet3DLight ===")
    model_light = AERBUNet3DLight()
    y_light = model_light(x)
    print(f"输出形状: {y_light.shape}")
    
    # 计算参数量
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    total_full, trainable_full = count_parameters(model_full)
    total_light, trainable_light = count_parameters(model_light)
    
    print(f"\n参数量统计:")
    print(f"完整版: 总参数量 = {total_full:,}, 可训练参数 = {trainable_full:,}")
    print(f"轻量版: 总参数量 = {total_light:,}, 可训练参数 = {trainable_light:,}")
    print(f"轻量版参数减少: {(1 - total_light/total_full)*100:.1f}%")
    
    # 测试内存占用
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"\nGPU内存测试:")
        for name, model in [("完整版", model_full), ("轻量版", model_light)]:
            model = model.cuda()
            x_gpu = x.cuda()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x_gpu)
            
            memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{name} GPU峰值内存: {memory:.1f} MB")
            
            model = model.cpu()
            torch.cuda.empty_cache()

# 简化的3D UNet版本（更轻量）
class SimpleAERBUNet3D(nn.Module):
    """最简化的3D增强UNet，2层下采样，用于快速测试"""
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=0.1):
        super().__init__()
        
        # 编码器
        self.enc1 = LightResidualBlock3D(in_channels, 16, dropout_prob, use_attention=True)
        self.enc2 = LightResidualBlock3D(16, 32, dropout_prob, use_attention=True)
        self.pool = nn.MaxPool3d(2)
        
        # 瓶颈层
        self.bottleneck = LightResidualBlock3D(32, 64, dropout_prob, use_attention=True)
        
        # 解码器
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = LightResidualBlock3D(64, 32, dropout_prob, use_attention=True)
        
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec2 = LightResidualBlock3D(32, 16, dropout_prob, use_attention=True)
        
        # 输出层
        self.out = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)                     # [B, 16, 128, 128, 128]
        e2 = self.enc2(self.pool(e1))         # [B, 32, 64, 64, 64]
        
        # 瓶颈层
        bn = self.bottleneck(self.pool(e2))   # [B, 64, 32, 32, 32]
        
        # 解码器
        d1 = self.up1(bn)                     # [B, 32, 64, 64, 64]
        d1 = torch.cat([e2, d1], dim=1)       # [B, 64, 64, 64, 64]
        d1 = self.dec1(d1)                    # [B, 32, 64, 64, 64]
        
        d2 = self.up2(d1)                     # [B, 16, 128, 128, 128]
        d2 = torch.cat([e1, d2], dim=1)       # [B, 32, 128, 128, 128]
        d2 = self.dec2(d2)                    # [B, 16, 128, 128, 128]
        
        return self.out(d2)

if __name__ == "__main__":
    print("=" * 50)
    print("3D Attention Residual UNet (仅主输出)")
    print("=" * 50)
    test_models()
    
    # 测试最简化版本
    print("\n=== 最简化版 SimpleAERBUNet3D ===")
    x = torch.randn(1, 1, 128, 128, 128)
    model_simple = SimpleAERBUNet3D()
    y_simple = model_simple(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_simple.shape}")


    # 计算参数量
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    total_simple, trainable_simple = count_parameters(model_simple)
    print(f"最简化版总参数量: {total_simple:,}, 可训练参数 = {trainable_simple:,}")