"""Segmentation (SegFormer-B0, ADE20K, 150 classes) and depth (MiDaS DPT-Hybrid) estimators that give MURL its
semantic and geometric inputs. `forward` gives the hard maps used for training data and scoring; `soft` keeps the
gradient to the image for the MURL loss."""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

SEGFORMER = "nvidia/segformer-b0-finetuned-ade-512-512"
DPT_HYBRID = "Intel/dpt-hybrid-midas"


class SegFormerLabels(nn.Module):
    """Class labels at the input resolution (logits upsampled before the argmax). Inputs above `max_pixels` run
    downscaled and their labels are upsampled (nearest)."""

    def __init__(self, name=SEGFORMER, max_pixels=4_000_000):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(name).eval().requires_grad_(False)
        self.max_pixels = max_pixels
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def probs(self, rgb):
        logits = self.model(pixel_values=(rgb - self.mean) / self.std).logits
        return F.interpolate(logits.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False).softmax(1)

    @torch.no_grad()
    def forward(self, rgb):
        """rgb (B, 3, H, W) in [0, 1] -> (B, H, W) uint8 labels."""
        H, W = rgb.shape[-2:]
        x = rgb
        if H * W > self.max_pixels:
            s = (self.max_pixels / (H * W)) ** 0.5
            x = F.interpolate(rgb, size=(int(H * s), int(W * s)), mode="bilinear", align_corners=False, antialias=True)
        logits = self.model(pixel_values=(x - self.mean) / self.std).logits
        labels = F.interpolate(logits.float(), size=x.shape[-2:], mode="bilinear", align_corners=False).argmax(1, keepdim=True)
        if labels.shape[-2:] != (H, W):
            labels = F.interpolate(labels.float(), size=(H, W), mode="nearest").long()
        return labels[:, 0].to(torch.uint8)

    def soft(self, rgb):
        """Straight-through one-hot (B, 150, H, W): the argmax one-hot forward, the softmax gradient backward."""
        p = self.probs(rgb)
        hard = F.one_hot(p.argmax(1), p.shape[1]).permute(0, 3, 1, 2).to(p.dtype)
        return hard + p - p.detach()


class MiDaSDepth(nn.Module):
    """Relative inverse depth in [0, 1] (per-image min-max) at the input resolution, with the MiDaS preprocessing:
    aspect ratio kept, short side `infer_size`, sides a multiple of 32, mean and std 0.5."""

    def __init__(self, name=DPT_HYBRID, infer_size=384):
        super().__init__()
        from transformers import DPTForDepthEstimation

        self.model = DPTForDepthEstimation.from_pretrained(name).eval().requires_grad_(False)
        emb = self.model.dpt.embeddings
        emb.forward = functools.partial(emb.forward, interpolate_pos_encoding=True)
        self.patch = self.model.config.patch_size
        self.infer_size = infer_size

    def predict(self, x):
        # DPTForDepthEstimation.forward with the true patch grid, so non-square inputs work with the hybrid backbone
        m = self.model
        out = m.dpt(x, output_hidden_states=True)
        feats = list(out.intermediate_activations)
        feats += [f for i, f in enumerate(out.hidden_states[1:]) if i in m.config.backbone_out_indices[2:]]
        return m.head(m.neck(feats, x.shape[-2] // self.patch, x.shape[-1] // self.patch))

    def soft(self, rgb):
        H, W = rgb.shape[-2:]
        s = self.infer_size / min(H, W)
        size = (max(32, round(H * s / 32) * 32), max(32, round(W * s / 32) * 32))
        x = F.interpolate(rgb, size=size, mode="bicubic", align_corners=False)
        pred = self.predict((x.clamp(0, 1) - 0.5) / 0.5).unsqueeze(1).float()
        pred = F.interpolate(pred, size=(H, W), mode="bicubic", align_corners=False)
        lo, hi = pred.amin((1, 2, 3), keepdim=True), pred.amax((1, 2, 3), keepdim=True)
        return (pred - lo) / (hi - lo + 1e-8)

    @torch.no_grad()
    def forward(self, rgb):
        """rgb (B, 3, H, W) in [0, 1] -> (B, 1, H, W) float in [0, 1]."""
        return self.soft(rgb)
