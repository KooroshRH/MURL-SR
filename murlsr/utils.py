import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path, overrides=()):
    """YAML config plus `a.b.c=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def seed_of(*keys):
    """Deterministic 63-bit seed from any keys."""
    return int.from_bytes(hashlib.sha256("/".join(map(str, keys)).encode()).digest()[:8], "little") % 2**63


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def init_distributed():
    """(rank, world size, device); works with plain `python` and with `torchrun`."""
    world, rank, local = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def all_reduce_mean(t):
    if dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return t


def all_reduce_sum(t):
    if dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t)
    return t


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


class Degradation:
    """VOSR's Real-ESRGAN degradation (its training pipeline), with the random state seeded per image. Needs basicsr."""

    def __init__(self, device):
        import torchvision.transforms._functional_tensor as ft

        sys.modules.setdefault("torchvision.transforms.functional_tensor", ft)  # basicsr 1.4.2 with torchvision >= 0.17
        from .third_party.vosr.realesrgan_gpu import RealESRGAN_degradation

        self.deg = RealESRGAN_degradation("params_realsr.yml", device=device)

    @torch.no_grad()
    def __call__(self, hr, seed):
        """hr (1, 3, H, W) in [0, 1] -> LQ in [0, 1], degraded x4 and resized back to H x W."""
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        return self.deg.degrade_process(hr, resize_bak=True)[1]
