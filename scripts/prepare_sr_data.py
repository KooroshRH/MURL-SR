"""LR-HR training pairs: one random 512 crop of every LSDIR training image as HR, its LR from VOSR's Real-ESRGAN
degradation (x4, resized back to 512), both seeded per image. Writes <data.root>/{hr,lq}/<id>.png. Resumable;
--shard / --num-shards split the images over jobs.

    python scripts/prepare_sr_data.py --config configs/sr.yaml
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.data import list_images, read_rgb, save_png, to_tensor, to_uint8  # noqa: E402
from murlsr.utils import Degradation, load_config, seed_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    d, S = cfg["data"], cfg["data"]["crop"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deg = Degradation(device)
    n = 0
    for rel in list_images(d["images"])[args.shard :: args.num_shards]:
        name = os.path.splitext(rel)[0].replace("/", "_")
        if os.path.exists(os.path.join(d["root"], "lq", name + ".png")):
            continue
        img = read_rgb(os.path.join(d["images"], rel))
        H, W = img.shape[:2]
        if min(H, W) < S:
            continue
        rng = np.random.default_rng(seed_of(d["seed"], "crop", name))
        t, l = int(rng.integers(H - S + 1)), int(rng.integers(W - S + 1))
        hr = img[t : t + S, l : l + S]
        lq = deg(to_tensor(hr, device), seed_of(d["seed"], "degrade", name))
        save_png(os.path.join(d["root"], "hr", name + ".png"), hr)
        save_png(os.path.join(d["root"], "lq", name + ".png"), to_uint8(lq)[0])
        n += 1
    print(f"shard {args.shard}/{args.num_shards}: {n} new pairs in {d['root']}", flush=True)


if __name__ == "__main__":
    main()
