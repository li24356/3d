import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """A simple 3D U-Net suitable for 128x128x128 volumes.

    Default: single-channel input, single-channel output (logits).
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(UNet3D, self).__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)

        self.bot = DoubleConv(f * 8, f * 16)

        self.up3 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up1 = Up(f * 4, f * 2)
        self.up0 = Up(f * 2, f)

        self.outc = OutConv(f, out_channels)

    def forward(self, x):
        x0 = self.inc(x)     # -> f, 128
        x1 = self.down1(x0)  # -> 2f, 64
        x2 = self.down2(x1)  # -> 4f, 32
        x3 = self.down3(x2)  # -> 8f, 16

        xb = self.bot(x3)    # -> 16f, 8

        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # quick shape test
    model = UNet3D(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 128, 128, 128)
    y = model(x)
    print('output shape', y.shape)
