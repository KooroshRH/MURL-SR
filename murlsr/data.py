import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None
EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def list_images(root):
    """Image paths under `root` (recursive), relative to it, sorted."""
    paths = []
    for d, _, files in os.walk(root):
        paths += [os.path.join(d, f) for f in files if f.lower().endswith(EXTENSIONS)]
    return sorted(os.path.relpath(p, root) for p in paths)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def save_png(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path)


def to_tensor(a, device):
    """uint8 (H, W, 3) or (N, H, W, 3) -> float (N, 3, H, W) in [0, 1]."""
    t = torch.from_numpy(np.array(a)).to(device).float() / 255
    return (t[None] if t.dim() == 3 else t).permute(0, 3, 1, 2)


def to_uint8(t):
    """float (N, 3, H, W) in [0, 1] -> uint8 (N, H, W, 3)."""
    return (t.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def read_index(maps_root):
    """Image ids that have precomputed maps (scripts/precompute_maps.py)."""
    ids = []
    for f in sorted(os.listdir(maps_root)):
        if f.startswith("index_") and f.endswith(".txt"):
            with open(os.path.join(maps_root, f)) as fh:
                ids += [line.strip() for line in fh if line.strip()]
    return sorted(set(ids))


class FullImages(Dataset):
    """Whole images as uint8 (3, H, W) with their id (path relative to root, without extension)."""

    def __init__(self, root, rel_paths):
        self.root, self.rel_paths = root, rel_paths

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        rel = self.rel_paths[i]
        img = read_rgb(os.path.join(self.root, rel))
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), os.path.splitext(rel)[0]


class MURLMaps(Dataset):
    """Images with their precomputed segmentation and depth maps, cropped on the fly: a random crop (train) or the
    center crop (val), the same box and flip for all three. Depth is re-normalized to [0, 1] over the crop.
    Returns rgb uint8 (3, S, S), seg uint8 (S, S), depth float (1, S, S)."""

    def __init__(self, images_root, maps_root, crop=512, train=True, hflip=True, limit=None):
        self.images_root, self.maps_root = images_root, maps_root
        self.crop, self.train, self.hflip = crop, train, hflip
        self.ids = read_index(maps_root)[:limit]
        if not self.ids:
            raise FileNotFoundError(f"no indexed images under {maps_root}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        rgb = read_rgb(os.path.join(self.images_root, name + ".png"))
        seg = np.asarray(Image.open(os.path.join(self.maps_root, "seg", name + ".png")))
        depth = np.asarray(Image.open(os.path.join(self.maps_root, "depth", name + ".png")))
        H, W = seg.shape
        assert rgb.shape[:2] == (H, W) == depth.shape, f"size mismatch for {name}"
        S = self.crop
        if self.train:
            t, l = int(torch.randint(0, H - S + 1, ())), int(torch.randint(0, W - S + 1, ()))
        else:
            t, l = (H - S) // 2, (W - S) // 2
        rgb, seg, depth = rgb[t:t + S, l:l + S], seg[t:t + S, l:l + S], depth[t:t + S, l:l + S]
        if self.train and self.hflip and torch.rand(()) < 0.5:
            rgb, seg, depth = rgb[:, ::-1], seg[:, ::-1], depth[:, ::-1]
        depth = depth.astype(np.float32) / 65535
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(seg)),
                torch.from_numpy(np.ascontiguousarray(depth))[None])
