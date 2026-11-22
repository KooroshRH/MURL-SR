import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    """Concatenates the features of the other modalities and projects them with a 1x1 conv, BN and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, *feats):
        return self.act(self.bn(self.conv(torch.cat(feats, dim=1))))


class CrossAttention(nn.Module):
    """Pre-norm cross-attention over spatial tokens: x + W_o softmax(Q K^T / sqrt(d)) V, with queries from x and keys and
    values from ctx (average-pooled by kv_pool first)."""

    def __init__(self, dim=128, heads=4, kv_pool=1):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.kv_pool = heads, kv_pool
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, ctx):
        B, C, H, W = x.shape
        if self.kv_pool > 1:
            ctx = F.avg_pool2d(ctx, self.kv_pool)
        q = self.q(self.norm_q(x.flatten(2).transpose(1, 2)))
        k, v = self.kv(self.norm_kv(ctx.flatten(2).transpose(1, 2))).chunk(2, dim=-1)
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v))
        a = self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
        return x + a.transpose(1, 2).reshape(B, C, H, W)
