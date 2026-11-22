"""MURL-SR training (Eq. 23): LoRA fine-tuning of the one-step VOSR with hallucination-aware flow DPO on the pairs of
scripts/make_pairs.py plus direct MURL supervision of the one-step output. Single GPU or torchrun; train.batch pairs
per step over all GPUs, with gradient accumulation. A rerun resumes from <output>/checkpoints/last.pt; at the end the
LoRA is merged into a VOSR-format checkpoint in <output>/final (also: scripts/export_checkpoint.py).

    python scripts/train_sr.py --config configs/sr.yaml [key.sub=value ...]
    torchrun --nproc_per_node 4 scripts/train_sr.py --config configs/sr.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.data import read_rgb  # noqa: E402
from murlsr.murl import MURLScore  # noqa: E402
from murlsr.utils import cleanup, init_distributed, load_config, seed_of  # noqa: E402
from murlsr.vosr import VOSR, export, load_lora  # noqa: E402


class Pairs(Dataset):
    def __init__(self, root):
        self.root = root
        self.names = sorted(f[:-4] for f in os.listdir(os.path.join(root, "pairs")) if f.endswith(".npz"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        with np.load(os.path.join(self.root, "pairs", name + ".npz")) as p:
            zw, zl = torch.from_numpy(p["winner"]), torch.from_numpy(p["loser"])
        lq, hr = (torch.from_numpy(read_rgb(os.path.join(self.root, s, name + ".png")).copy()) for s in ("lq", "hr"))
        return lq, hr, zw, zl


def stack_cond(cond, *idx):
    """The condition repeated for the given index lists, concatenated (DPO rows, then one-step rows)."""
    return {"lq": torch.cat([cond["lq"][i] for i in idx]), "venc": [torch.cat([f[i] for i in idx]) for f in cond["venc"]]}


def micro_loss(t, vosr, score, lq, hr, zw, zl):
    """Preference loss (Eq. 20 to 22, with DPO-Positive) and MURL loss (Eq. 17) of one micro-batch."""
    B, dev = zw.shape[0], zw.device
    with torch.no_grad():
        cond = vosr.condition(lq)
    # Flow DPO on the jump to the clean latent: same t and noise for the preferred and rejected sample
    s = torch.rand(B, device=dev).view(-1, 1, 1, 1)
    eps = torch.randn_like(zw)
    zt = torch.cat([(1 - s) * zw + s * eps, (1 - s) * zl + s * eps])
    target = torch.cat([eps - zw, eps - zl])
    ts, rs = torch.cat([s.flatten(), s.flatten()]), torch.zeros(2 * B, device=dev)
    # Direct MURL supervision of the deployed one-step output from fresh noise
    nj = min(B, t["murl_items"])
    eps_j = torch.randn_like(zw[:nj])
    rows, jrows = list(range(B)) * 2, list(range(nj))
    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad(), vosr.reference():
            pred_ref = vosr.velocity(zt, ts, stack_cond(cond, rows), rs)
        pred = vosr.velocity(torch.cat([zt, eps_j]), torch.cat([ts, torch.ones(nj, device=dev)]), stack_cond(cond, rows, jrows),
                             torch.zeros(2 * B + nj, device=dev))
    pred, u_j = pred[: 2 * B], pred[2 * B :]
    e_w, e_l = (pred.float() - target).pow(2).mean((1, 2, 3)).split(B)
    r_w, r_l = (pred_ref.float() - target).pow(2).mean((1, 2, 3)).split(B)
    inside = -t["beta"] * ((e_w - r_w) - (e_l - r_l) + t["dpop"] * F.relu(e_w - r_w))
    l_pref = -F.logsigmoid(inside).mean()
    x_hat = vosr.decode(eps_j - u_j.float()) * 0.5 + 0.5
    l_murl = score.loss(x_hat, hr[:nj] * 0.5 + 0.5)
    loss = t["lambda_pref"] * l_pref + t["lambda_murl"] * l_murl
    return loss, {"loss": loss.detach(), "pref": l_pref.detach(), "murl": l_murl.detach(), "acc": (inside > 0).float().mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    t = cfg["train"]
    rank, world, dev = init_distributed()
    main0 = rank == 0
    accum, rem = divmod(t["batch"], t["micro_batch"] * world)
    assert accum >= 1 and rem == 0, f"train.batch {t['batch']} is not a multiple of micro_batch x {world} GPUs"
    ckdir = os.path.join(t["output"], "checkpoints")
    last = os.path.join(ckdir, "last.pt")
    os.makedirs(ckdir, exist_ok=True)

    torch.manual_seed(cfg["data"]["seed"])  # the same LoRA initialization on every rank
    vosr = VOSR(cfg["vosr"]["checkpoint"], cfg["vosr"]["vae"], dev)
    params = vosr.add_lora(t["lora"]["rank"], t["lora"]["alpha"])
    vosr.gradient_checkpointing()
    score = MURLScore(cfg["murl"]).to(dev)
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=0.0)
    step = 0
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        load_lora(vosr.dit, ck["lora"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    torch.manual_seed(seed_of(cfg["data"]["seed"], "rank", rank, step))

    data = Pairs(cfg["data"]["root"])
    sampler = DistributedSampler(data, world, rank, seed=cfg["data"]["seed"], drop_last=True) if world > 1 else \
        RandomSampler(data, generator=torch.Generator().manual_seed(seed_of(cfg["data"]["seed"], "order", step)))
    loader = DataLoader(data, t["micro_batch"], sampler=sampler, drop_last=True, num_workers=t["num_workers"],
                        pin_memory=True, persistent_workers=t["num_workers"] > 0)
    if main0:
        print(f"{len(data)} pairs, {sum(p.numel() for p in params) / 1e6:.1f}M LoRA parameters, {world} GPU(s), "
              f"accumulation {accum}, start step {step}", flush=True)

    def save(path):
        if main0:
            torch.save({"step": step, "lora": vosr.lora_state_dict(), "opt": opt.state_dict(), "cfg": cfg}, path + ".tmp")
            os.replace(path + ".tmp", path)

    def batches():
        epoch = step
        while True:
            if world > 1:
                sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    it, logs, t0 = batches(), {}, time.time()
    while step < t["steps"]:
        for _ in range(accum):
            lq, hr, zw, zl = next(it)
            lq, hr = (x.to(dev).permute(0, 3, 1, 2).float() / 255 * 2 - 1 for x in (lq, hr))
            loss, rec = micro_loss(t, vosr, score, lq, hr, zw.to(dev), zl.to(dev))
            (loss / accum).backward()
            for k, v in rec.items():
                logs[k] = logs.get(k, 0.0) + float(v) / accum
        if world > 1:
            flat = torch.cat([p.grad.flatten() for p in params])
            dist.all_reduce(flat)
            flat /= world
            for p, g in zip(params, flat.split([p.numel() for p in params])):
                p.grad.copy_(g.view_as(p))
        torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if main0 and step % t.get("log_every", 50) == 0:
            n = t.get("log_every", 50)
            print(json.dumps({"step": step, **{k: round(v / n, 5) for k, v in logs.items()},
                              "sec_per_step": round((time.time() - t0) / n, 3)}), flush=True)
            logs, t0 = {}, time.time()
        elif not main0:
            logs = {}
        if step % t["ckpt_every"] == 0:
            save(last)
    save(last)
    if main0:
        export(cfg["vosr"]["checkpoint"], vosr.lora_state_dict(), t["lora"]["rank"], t["lora"]["alpha"],
               os.path.join(t["output"], "final"))
        print(f"VOSR-format checkpoint: {os.path.join(t['output'], 'final')}", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
