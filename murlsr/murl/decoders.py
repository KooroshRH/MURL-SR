import torch
import torch.nn as nn

from .blocks import ResBlock, group_norm
from .encoders import LATENT_CHANNELS


class ResDecoder(nn.Module):
    """Residual decoder in the Stable Diffusion VAE layout, without encoder skips: 3x3 conv, four levels of ResBlocks
    (H/8 to H) with nearest x2 upsampling and a conv between levels, then GroupNorm, SiLU and a conv out."""

    def __init__(self, out_channels, channels=(128, 128, 64, 32), blocks=(2, 2, 1, 1), latent_stride=8, sigmoid=True,
                 grad_checkpoint=False):
        super().__init__()
        assert len(channels) == len(blocks) == 4 and latent_stride in (4, 8)
        n_up = latent_stride.bit_length() - 1
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, channels[0], 3, padding=1)
        levels = []
        for i, (c, n) in enumerate(zip(channels, blocks)):
            ups = [nn.Upsample(scale_factor=2, mode="nearest")] if i > 3 - n_up else []
            up = [] if i == 0 else [*ups, nn.Conv2d(channels[i - 1], c, 3, padding=1)]
            levels.append(nn.Sequential(*up, *[ResBlock(c, c, grad_checkpoint) for _ in range(n)]))
        self.levels = nn.Sequential(*levels)
        self.out = nn.Sequential(group_norm(channels[-1]), nn.SiLU(), nn.Conv2d(channels[-1], out_channels, 3, padding=1))
        self.sigmoid = sigmoid

    def forward(self, z):
        x = self.out(self.levels(self.conv_in(z)))
        return torch.sigmoid(x) if self.sigmoid else x
