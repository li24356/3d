import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 内部注意力组件 (CBAM) - 用于增强特征提取
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 使用1x1x1卷积代替全连接层，减少参数并保留空间信息
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock3D(nn.Module):
    """结合通道和空间注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ==========================================
# 2. 外部注意力组件 (Attention Gate) - 用于跳跃连接
# ==========================================
class AttentionGate3D(nn.Module):
    """
    g: decoder feature
    x: encoder feature (skip connection)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ==========================================
# 3. 基础卷积块 (Residual + InstanceNorm + CBAM)
# ==========================================
class ResBlock3D(nn.Module):
    """
    重型残差块：
    Conv -> IN -> ReLU -> Conv -> IN -> CBAM -> Add -> ReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_ch) # 使用 InstanceNorm
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_ch)
        
        self.cbam = CBAMBlock3D(out_ch) # 块内注意力
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.InstanceNorm3d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # 应用CBAM注意力优化特征
        out = self.cbam(out)
        
        out += residual
        return F.relu(out, inplace=True)

# ==========================================
# 4. 上采样模块
# ==========================================
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.att_gate = AttentionGate3D(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2)
        self.conv = ResBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder, x2: encoder skip
        x1 = self.up(x1)
        
        # 自动对齐尺寸（防止奇数尺寸报错）
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        if diffZ != 0 or diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])
        
        # Gate过滤Skip特征
        x2 = self.att_gate(g=x1, x=x2)
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ==========================================
# 5. 最终模型: High-Performance Residual Attention UNet
# ==========================================
class AERBPRO(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        """
        base_channels: 建议设为 32 或 64 以获得最佳性能 (显存允许的话)
        """
        super().__init__()
        
        # Encoder
        self.inc = ResBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ResBlock3D(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ResBlock3D(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = ResBlock3D(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock3D(base_channels*8, base_channels*16)
        
        # Decoder
        self.up1 = UpBlock(base_channels*16, base_channels*8, base_channels*8)
        self.up2 = UpBlock(base_channels*8, base_channels*4, base_channels*4)
        self.up3 = UpBlock(base_channels*4, base_channels*2, base_channels*2)
        self.up4 = UpBlock(base_channels*2, base_channels, base_channels)
        
        # Output
        self.outc = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        
        # Bottleneck
        bn = self.bottleneck(self.pool4(x4))
        
        # Decoder Path (with Attention Gates)
        d1 = self.up1(bn, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        
        return self.outc(d4)

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    import time
    
    # 模拟输入：Batch=1 (模拟真实3D分割场景)
    x = torch.randn(1, 1, 128, 128, 128)
    
    print("=" * 50)
    print("H-ResAttUNet3D (High Performance Mode)")
    print("配置: Residual + InstanceNorm + CBAM + AttentionGate")
    print("=" * 50)
    
    # 1. 初始化模型 (Base=64 是比较均衡的高性能配置，显存够可以开64)
    model = AERBPRO(in_channels=1, out_channels=1, base_channels=64)
    
    # 2. 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.2f} M (Million)")
    
    # 3. 运行测试
    model.eval()
    with torch.no_grad():
        start = time.time()
        y = model(x)
        end = time.time()
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"推理耗时: {end - start:.4f}s")
    
    # 4. 显存压力测试
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        torch.cuda.reset_peak_memory_stats()
        try:
            y = model(x)
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"GPU 显存占用: {mem:.2f} GB")
        except RuntimeError as e:
            print("显存不足，请降低 base_channels 或 patch size")