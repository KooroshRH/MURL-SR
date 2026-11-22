import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .blocks import ResBlock, group_norm

LATENT_CHANNELS = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class ResNetEncoder(nn.Module):
    """ImageNet ResNet-18 stem and its first two residual stages (segmentation and depth): 128 channels at H/8, or at
    H/4 with latent_stride=4 (layer2 at stride 1). The first convolution is adapted to the input channels."""

    def __init__(self, in_channels, latent_stride=8, pretrained=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels != 3:
            w = base.conv1.weight.data
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if in_channels == 1:
                    base.conv1.weight.copy_(w.sum(1, keepdim=True))
                else:
                    base.conv1.weight.copy_(w.mean(1, keepdim=True).repeat(1, in_channels, 1, 1))
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        assert latent_stride in (4, 8), latent_stride
        if latent_stride == 4:
            self.layer2[0].conv1.stride = (1, 1)
            self.layer2[0].downsample[0].stride = (1, 1)

    def forward(self, x):
        return self.layer2(self.layer1(self.pool(self.layer0(x))))


class ResEncoder(nn.Module):
    """Multi-level residual encoder trained from scratch (visual): 3x3 conv, four levels of ResBlocks with a stride-2
    conv between levels, then GroupNorm, SiLU and a conv to 128 channels at H/latent_stride."""

    def __init__(self, in_channels=3, channels=(64, 128, 256, 256), blocks=(1, 1, 2, 2), latent_stride=8,
                 grad_checkpoint=False, imagenet_norm=True):
        super().__init__()
        assert len(channels) == len(blocks) == 4 and latent_stride in (4, 8)
        n_down = latent_stride.bit_length() - 1
        self.norm = ImageNetNorm() if imagenet_norm else nn.Identity()
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        levels = []
        for i, (c, n) in enumerate(zip(channels, blocks)):
            down = [] if i == 0 else [nn.Conv2d(channels[i - 1], c, 3, stride=2 if i <= n_down else 1, padding=1)]
            levels.append(nn.Sequential(*down, *[ResBlock(c, c, grad_checkpoint) for _ in range(n)]))
        self.levels = nn.Sequential(*levels)
        self.out = nn.Sequential(group_norm(channels[-1]), nn.SiLU(), nn.Conv2d(channels[-1], LATENT_CHANNELS, 3, padding=1))

    def forward(self, x):
        return self.out(self.levels(self.conv_in(self.norm(x))))
