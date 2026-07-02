"""
model.py - CNN architectures for the closure NN.

Two options provided:
  - PeriodicConvNet:  simple stack of periodic-padded Conv2d blocks. Good baseline.
  - PeriodicUNet:     small U-Net with periodic boundaries. Captures multi-scale
                      structure (closure depends on both small-scale and
                      large-scale features).

Both take input (B, C_in, Ny, Nx) and produce (B, 1, Ny, Nx) -- the predicted
forcing field f_NN. Domain is periodic in both x and y, so we use circular
padding everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _periodic_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Circular pad by p on the last two dims."""
    return F.pad(x, (p, p, p, p), mode='circular')


class PeriodicConvBlock(nn.Module):
    """Conv2d (with circular padding) -> GroupNorm -> SiLU."""
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, groups: int = 8):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=0, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = (self.kernel - 1) // 2
        x = _periodic_pad(x, p)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PeriodicConvNet(nn.Module):
    """Stack of N PeriodicConvBlocks, then a 1x1 Conv2d to a single output channel.

    Args:
        in_channels:    number of input channels (e.g. 1 for omega_0 only,
                        2 for (omega_0, psi_0), etc.)
        hidden_channels: width of the conv layers
        depth:          number of conv blocks
        kernel:         kernel size of conv layers
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64,
                 depth: int = 6, kernel: int = 5):
        super().__init__()
        layers = [PeriodicConvBlock(in_channels, hidden_channels, kernel=kernel)]
        for _ in range(depth - 1):
            layers.append(PeriodicConvBlock(hidden_channels, hidden_channels,
                                            kernel=kernel))
        self.body = nn.Sequential(*layers)
        # 1x1 head -> 1 output channel (f_NN)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.head(h)


class _DownBlock(nn.Module):
    """PeriodicConvBlock + AvgPool2d for U-Net downsampling."""
    def __init__(self, in_c: int, out_c: int, kernel: int = 3):
        super().__init__()
        self.conv1 = PeriodicConvBlock(in_c, out_c, kernel=kernel)
        self.conv2 = PeriodicConvBlock(out_c, out_c, kernel=kernel)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv2(self.conv1(x))
        return self.pool(skip), skip


class _UpBlock(nn.Module):
    """Bilinear upsample + concat skip + PeriodicConvBlock x2."""
    def __init__(self, in_c: int, skip_c: int, out_c: int, kernel: int = 3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = PeriodicConvBlock(in_c + skip_c, out_c, kernel=kernel)
        self.conv2 = PeriodicConvBlock(out_c, out_c, kernel=kernel)

    def forward(self, x, skip):
        x = self.up(x)
        # Periodic upsample of skip should match shape; just concat.
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class PeriodicUNet(nn.Module):
    """U-Net with 2 downsamples + bottleneck + 2 upsamples. Periodic padding.

    Channel widths: base_channels, 2*base_channels, 4*base_channels.
    For 256x256 input, intermediate features are at 128x128 and 64x64.

    Args:
        in_channels:    number of input channels
        base_channels:  width of the first (highest-resolution) layer
        kernel:         spatial kernel size of all conv layers
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32, kernel: int = 3):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels*2, base_channels*4

        self.in_block = nn.Sequential(
            PeriodicConvBlock(in_channels, c1, kernel=kernel),
            PeriodicConvBlock(c1, c1, kernel=kernel),
        )
        self.down1 = _DownBlock(c1, c2, kernel=kernel)
        self.down2 = _DownBlock(c2, c3, kernel=kernel)

        self.bottleneck = nn.Sequential(
            PeriodicConvBlock(c3, c3, kernel=kernel),
            PeriodicConvBlock(c3, c3, kernel=kernel),
        )

        self.up2 = _UpBlock(c3, c3, c2, kernel=kernel)
        self.up1 = _UpBlock(c2, c2, c1, kernel=kernel)
        self.head = nn.Conv2d(c1, 1, kernel_size=1, bias=True)

    def forward(self, x):
        s0 = self.in_block(x)              # (c1, H,   W)
        x1, s1 = self.down1(s0)            # x1: (c2, H/2, W/2);  s1: (c2, H,   W)
        x2, s2 = self.down2(x1)            # x2: (c3, H/4, W/4);  s2: (c3, H/2, W/2)
        b  = self.bottleneck(x2)           # (c3, H/4, W/4)
        u2 = self.up2(b,  s2)              # up: (c3, H/2, W/2), cat s2(c3): (2c3, H/2, W/2) -> conv -> (c2, H/2, W/2)
        u1 = self.up1(u2, s1)              # up: (c2, H,   W),   cat s1(c2): (2c2, H,   W)   -> conv -> (c1, H,   W)
        return self.head(u1)


def build_model(name: str, in_channels: int = 1, **kwargs) -> nn.Module:
    """Factory: 'cnn', 'unet'."""
    name = name.lower()
    if name in ('cnn', 'periodicconvnet', 'periodic_cnn'):
        return PeriodicConvNet(in_channels=in_channels, **kwargs)
    if name in ('unet', 'periodicunet', 'periodic_unet'):
        return PeriodicUNet(in_channels=in_channels, **kwargs)
    raise ValueError(f"unknown model name '{name}'")


if __name__ == '__main__':
    import torch
    print("CNN:")
    m = build_model('cnn', in_channels=2)
    x = torch.randn(1, 2, 256, 256)
    y = m(x)
    print(f"  in: {tuple(x.shape)}, out: {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in m.parameters()):,}")

    print("UNet:")
    m = build_model('unet', in_channels=2)
    y = m(x)
    print(f"  in: {tuple(x.shape)}, out: {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in m.parameters()):,}")