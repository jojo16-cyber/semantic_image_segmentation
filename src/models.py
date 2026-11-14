"""
Neural Network Architectures for Semantic Segmentation:
1. BaselineUNet: Canonical U-Net encoder-decoder architecture with standard 3x3 convolutions.
2. DilatedUNet: Controlled ablation introducing Dilated (Atrous) Convolutions in the bottleneck
   to expand the receptive field without adding parameters or altering spatial dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DoubleConv(nn.Module):
    """
    Standard double convolution block:
    [Conv2d(3x3) -> BatchNorm -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation  # preserves spatial resolution: (K-1)*d/2 = (3-1)*d/2 = d
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool then DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upscaling via Transposed Conv followed by feature concatenation and DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dilation=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 spatial dimensions
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """
    Standard U-Net Architecture (Ronneberger et al., 2015).
    4-level encoder, standard dense bottleneck, 4-level decoder with skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        # Encoder
        self.inc = DoubleConv(in_channels, c, dilation=1)          # 32
        self.down1 = DownBlock(c, c * 2)                           # 64
        self.down2 = DownBlock(c * 2, c * 4)                       # 128
        self.down3 = DownBlock(c * 4, c * 8)                       # 256

        # Standard Bottleneck (dilation = 1)
        self.bottleneck = DoubleConv(c * 8, c * 16, dilation=1)    # 512

        # Decoder
        self.up1 = UpBlock(c * 16, c * 8)                          # 256
        self.up2 = UpBlock(c * 8, c * 4)                           # 128
        self.up3 = UpBlock(c * 4, c * 2)                           # 64
        self.up4 = UpBlock(c * 2, c)                               # 32

        # Final Segmentation Head
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass with skip feature retention
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x2 if False else x3)  # x3 -> down3 -> x4

        # Bottleneck pass
        b = self.bottleneck(x4)

        # Decoder pass with skip concatenation
        d1 = self.up1(b, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        logits = self.outc(d4)
        return logits


class DilatedBottleneck(nn.Module):
    """
    Controlled Ablation Module:
    Replaces standard 3x3 convolutions with Dilated (Atrous) Convolutions (dilation rates = 2 and 4).
    Expands receptive field exponentially without losing spatial resolution or adding parameters.
    """
    def __init__(self, in_channels: int, out_channels: int, d1: int = 2, d2: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=d1,
            dilation=d1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=d2,
            dilation=d2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class DilatedUNet(nn.Module):
    """
    Ablation Architecture:
    Identical to BaselineUNet in all components EXCEPT the bottleneck,
    where standard convolutions are replaced with multi-scale Dilated Convolutions.
    Maintains exact parameter parity: Delta Parameters = 0.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        # Encoder (Identical to Baseline)
        self.inc = DoubleConv(in_channels, c, dilation=1)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)

        # Modified Bottleneck (Dilated Convolutions: dilation rates 2 and 4)
        self.bottleneck = DilatedBottleneck(c * 8, c * 16, d1=2, d2=4)

        # Decoder (Identical to Baseline)
        self.up1 = UpBlock(c * 16, c * 8)
        self.up2 = UpBlock(c * 8, c * 4)
        self.up3 = UpBlock(c * 4, c * 2)
        self.up4 = UpBlock(c * 2, c)

        # Final Segmentation Head (Identical)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        d1 = self.up1(b, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        logits = self.outc(d4)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name: str, num_classes: int = 3, base_channels: int = 32) -> nn.Module:
    """Factory function to instantiate models."""
    name_lower = model_name.lower().replace("-", "").replace("_", "")
    if "dilated" in name_lower:
        return DilatedUNet(num_classes=num_classes, base_channels=base_channels)
    elif "baseline" in name_lower or "unet" in name_lower:
        return BaselineUNet(num_classes=num_classes, base_channels=base_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'baseline' or 'dilated'.")
