import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def group_norm(channels):
    return nn.GroupNorm(min(32, channels // 4), channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.body = nn.Sequential(
            group_norm(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1),
            group_norm(out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        if self.grad_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        return self.skip(x) + self.body(x)
