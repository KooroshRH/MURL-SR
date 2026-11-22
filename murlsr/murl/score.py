"""MURL hallucination measure (Eq. 12 to 16) between an SR image x_hat and its ground truth x:
  H_pre  = sum_m mean_r [1 - cos(u_m(x_hat, r), u_m(x, r))]
  H_post = sum_m mean_r [1 - cos(q_m(x_hat, r), q_m(x, r))]
  H_rel  = sum_{m<n} mean_r |cos(q_m(x_hat, r), q_n(x_hat, r)) - cos(q_m(x, r), q_n(x, r))|
  H_MURL = H_pre + H_post + H_rel
u: unified representation before quantization, q: after, r: spatial location, cos over channels."""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modalities import MiDaSDepth, SegFormerLabels
from .model import MODALITIES, MURL, seg_onehot

PAIRS = tuple(itertools.combinations(MODALITIES, 2))


def hallucination(a, b):
    """a, b: (u, q) feature dicts of x_hat (batch N) and x (batch N or 1). Returns H_MURL per sample (N,)."""
    (ua, qa), (ub, qb) = a, b
    cos = lambda p, r: F.cosine_similarity(p, r, dim=1)  # noqa: E731
    h = 0
    for m in MODALITIES:
        h = h + (1 - cos(ua[m], ub[m])).mean((1, 2)) + (1 - cos(qa[m], qb[m])).mean((1, 2))
    for m, n in PAIRS:
        h = h + (cos(qa[m], qa[n]) - cos(qb[m], qb[n])).abs().mean((1, 2))
    return h


class MURLScore(nn.Module):
    """Frozen stage 2 MURL with the segmentation and depth estimators. Images are (B, 3, H, W) in [0, 1]."""

    def __init__(self, checkpoint):
        super().__init__()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert ck["cfg"]["stage"] == 2, f"{checkpoint} is not a stage 2 checkpoint"
        self.murl = MURL(ck["cfg"]["model"], pretrained=False)
        self.murl.load_state_dict(ck.get("ema") or ck["model"])
        self.seg, self.depth = SegFormerLabels(), MiDaSDepth()
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def features(self, rgb):
        """Features of the hard inputs (8-bit image, label map, 16-bit depth), as in MURL training."""
        vis = (rgb * 255).round() / 255
        depth = (self.depth(vis) * 65535).round() / 65535
        inputs = {"vis": vis.contiguous(memory_format=torch.channels_last), "seg": seg_onehot(self.seg(vis).long()),
                  "depth": depth.contiguous(memory_format=torch.channels_last)}
        return self.murl.features(inputs)

    def soft_features(self, rgb):
        """Features differentiable in rgb: straight-through segmentation one-hot and continuous depth."""
        inputs = {"vis": rgb.contiguous(memory_format=torch.channels_last), "seg": self.seg.soft(rgb), "depth": self.depth.soft(rgb)}
        return self.murl.features(inputs)

    @torch.no_grad()
    def forward(self, x_hat, x):
        """H_MURL of each x_hat (N) against x (N, or 1 shared)."""
        return hallucination(self.features(x_hat), self.features(x))

    def loss(self, x_hat, x):
        """Eq. 17: mean H_MURL over the batch, differentiable in x_hat."""
        return hallucination(self.soft_features(x_hat), self.features(x)).mean()
