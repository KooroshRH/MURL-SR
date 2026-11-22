"""Segmentation (SegFormer) and depth (MiDaS) maps of full images for MURL training; crops are taken at load time.
Writes <out>/seg/<id>.png (uint8 labels), <out>/depth/<id>.png (uint16, per-image min-max) and index_<shard>.txt.
Resumable; --shard / --num-shards split the images over jobs.

    python scripts/precompute_maps.py --src data/LSDIR/train --out data/maps/LSDIR/train
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.data import FullImages, list_images  # noqa: E402
from murlsr.modalities import MiDaSDepth, SegFormerLabels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="image folder (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(args.src)
    index_path = os.path.join(args.out, f"index_{args.shard:03d}.txt")
    done = set()
    if os.path.exists(index_path):
        with open(index_path) as f:
            done = {line.strip() for line in f if line.strip()}
    todo = [i for i in range(args.shard, len(paths), args.num_shards) if os.path.splitext(paths[i])[0] not in done]
    print(f"{len(paths)} images, shard {args.shard}/{args.num_shards}: {len(todo)} to do", flush=True)
    if not todo:
        return
    loader = DataLoader(Subset(FullImages(args.src, paths), todo), batch_size=None, num_workers=args.num_workers)
    seg_model, depth_model = SegFormerLabels().to(device), MiDaSDepth().to(device)
    os.makedirs(args.out, exist_ok=True)
    with open(index_path, "a") as index:
        for n, (img, name) in enumerate(loader, 1):
            rgb = img.to(device)[None].float() / 255
            seg = seg_model(rgb)[0].cpu().numpy()
            depth = (depth_model(rgb)[0, 0] * 65535).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)
            for sub, a in (("seg", seg), ("depth", depth)):
                os.makedirs(os.path.join(args.out, sub, os.path.dirname(name)), exist_ok=True)
                Image.fromarray(a).save(os.path.join(args.out, sub, name + ".png"))
            index.write(name + "\n")
            if n % 500 == 0:
                index.flush()
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
