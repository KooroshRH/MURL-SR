"""Hallucination-aware preference pairs (Eq. 18, 19): K one-step samples of the frozen VOSR per LR image, each from its
own seeded noise, scored with H_MURL against the HR; the lowest score is preferred, the highest rejected. Saves the
two latents (what DPO trains on) and all K scores to <data.root>/pairs/<id>.npz. Resumable; --shard / --num-shards
split the images over jobs.

    python scripts/make_pairs.py --config configs/sr.yaml
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.data import read_rgb, to_tensor  # noqa: E402
from murlsr.murl import MURLScore, hallucination  # noqa: E402
from murlsr.utils import load_config, seed_of  # noqa: E402
from murlsr.vosr import VOSR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    root, K, bs = cfg["data"]["root"], cfg["pairs"]["k"], cfg["pairs"]["score_batch"]
    device = "cuda"
    vosr = VOSR(cfg["vosr"]["checkpoint"], cfg["vosr"]["vae"], device)
    score = MURLScore(cfg["murl"]).to(device)
    names = sorted(f[:-4] for f in os.listdir(os.path.join(root, "lq")) if f.endswith(".png"))
    n, t0 = 0, time.time()
    for name in names[args.shard :: args.num_shards]:
        out = os.path.join(root, "pairs", name + ".npz")
        if os.path.exists(out):
            continue
        lq = to_tensor(read_rgb(os.path.join(root, "lq", name + ".png")), device) * 2 - 1
        hr = to_tensor(read_rgb(os.path.join(root, "hr", name + ".png")), device)
        cond = vosr.condition(lq)
        cond = {"lq": cond["lq"].expand(K, -1, -1, -1), "venc": [f.expand(K, -1, -1) for f in cond["venc"]]}
        g = torch.Generator(device).manual_seed(seed_of(cfg["data"]["seed"], name))
        with torch.no_grad():
            z = vosr.sample(cond, torch.randn(cond["lq"].shape, generator=g, device=device))
            ref = score.features(hr)
            h = torch.cat([hallucination(score.features(vosr.decode(z[i : i + bs]) * 0.5 + 0.5), ref) for i in range(0, K, bs)])
        w, l = int(h.argmin()), int(h.argmax())
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out + ".tmp.npz", winner=z[w].float().cpu().numpy(), loser=z[l].float().cpu().numpy(),
                 h_murl=h.cpu().numpy(), winner_index=w, loser_index=l)
        os.replace(out + ".tmp.npz", out)
        n += 1
        if n % 50 == 0:
            print(f"{n} pairs, {(time.time() - t0) / n:.1f} s each", flush=True)
    print(f"shard {args.shard}/{args.num_shards}: {n} new pairs in {os.path.join(root, 'pairs')}", flush=True)


if __name__ == "__main__":
    main()
