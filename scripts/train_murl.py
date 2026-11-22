"""Train MURL stage 1 (modality-specific autoencoders) or stage 2 (unified representation). Single GPU or torchrun.

    python scripts/train_murl.py --config configs/murl_stage1.yaml [key.sub=value ...]
    torchrun --nproc_per_node 4 scripts/train_murl.py --config configs/murl_stage2.yaml
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.data import MURLMaps  # noqa: E402
from murlsr.murl import MODALITIES, MURL, NUM_SEG_CLASSES, SpecificAEs, prepare_batch  # noqa: E402
from murlsr.murl.losses import orthogonality_loss, reconstruction_loss  # noqa: E402
from murlsr.utils import all_reduce_mean, all_reduce_sum, cleanup, init_distributed, load_config  # noqa: E402


def lr_at(step, t):
    """Linear warmup, then cosine decay from lr to min_lr at max_steps."""
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    p = min(1.0, (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"]))
    return t["min_lr"] + (t["lr"] - t["min_lr"]) * 0.5 * (1 + math.cos(math.pi * p))


def build(cfg):
    """The model to train and, for stage 2, the frozen stage 1 encoders."""
    if cfg["stage"] == 1:
        return SpecificAEs(cfg["model"]), None
    ck = torch.load(cfg["stage1"], map_location="cpu", weights_only=False)
    assert ck["cfg"]["stage"] == 1, f"{cfg['stage1']} is not a stage 1 checkpoint"
    spec = SpecificAEs(ck["cfg"]["model"], pretrained=False)
    spec.load_state_dict(ck.get("ema") or ck["model"])
    frozen = nn.ModuleDict({m: spec.aes[m].encoder for m in MODALITIES}).eval().requires_grad_(False)
    return MURL(cfg["model"]), frozen


def compute_loss(cfg, model, frozen, inputs, targets):
    w = cfg["loss"]
    out = model(inputs)
    logs = {f"rec/{m}": reconstruction_loss(m, out["recon"][m], targets[m]) for m in MODALITIES}
    total = w["rec"] * sum(logs.values())
    if frozen is not None:
        for m in MODALITIES:
            with torch.no_grad():
                spec = frozen[m](inputs[m])
            # unified features pooled to the stage 1 resolution
            uni = out["z"][m]
            uni = F.avg_pool2d(uni, uni.shape[-1] // spec.shape[-1]) if uni.shape[-1] != spec.shape[-1] else uni
            logs[f"vq/{m}"], logs[f"orth/{m}"] = out["vq_loss"][m], orthogonality_loss(spec, uni)
            total = total + w["vq"] * logs[f"vq/{m}"] + w["orth"] * logs[f"orth/{m}"]
    logs["total"] = total
    return total, {k: v.detach() for k, v in logs.items()}, out


@torch.no_grad()
def quality(out, targets):
    vis = (out["recon"]["vis"].float().clamp(0, 1) - targets["vis"]).pow(2).mean((1, 2, 3))
    pred, gt = out["recon"]["seg"].argmax(1), targets["seg"]
    inter = torch.stack([((pred == c) & (gt == c)).sum() for c in range(NUM_SEG_CLASSES)]).float()
    union = torch.stack([((pred == c) | (gt == c)).sum() for c in range(NUM_SEG_CLASSES)]).float()
    return {"psnr/vis": (-10 * torch.log10(vis.clamp_min(1e-10))).mean(), "acc/seg": (pred == gt).float().mean(),
            "miou/seg": (inter[union > 0] / union[union > 0]).mean(),
            "rmse/depth": (out["recon"]["depth"].float() - targets["depth"]).pow(2).mean().sqrt()}


@torch.no_grad()
def validate(cfg, model, frozen, loader, device, amp):
    model.eval()
    sums, n = {}, 0
    for b, batch in enumerate(loader):
        if b >= cfg["train"]["val_batches"]:
            break
        inputs, targets = prepare_batch(batch, device, torch.bfloat16)
        with amp():
            _, logs, out = compute_loss(cfg, model, frozen, inputs, targets)
        for k, v in {**logs, **quality(out, targets)}.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {f"val/{k}": v / max(n, 1) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    t, d = cfg["train"], cfg["data"]
    rank, world, device = init_distributed()
    main0 = rank == 0
    torch.manual_seed(cfg["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out_dir = cfg["output"]
    last = os.path.join(out_dir, "checkpoints", "last.pt")
    os.makedirs(os.path.dirname(last), exist_ok=True)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16)

    train_set = MURLMaps(d["train"]["images"], d["train"]["maps"], d["crop"], train=True)
    sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if world > 1 else None
    kw = dict(num_workers=d["num_workers"], pin_memory=True, persistent_workers=d["num_workers"] > 0)
    loader = DataLoader(train_set, d["batch_size"], sampler=sampler, shuffle=sampler is None, drop_last=True, **kw)
    val_loader = None
    if d.get("val") and main0:
        val_set = MURLMaps(d["val"]["images"], d["val"]["maps"], d["crop"], train=False)
        val_loader = DataLoader(val_set, d["batch_size"], **kw)

    raw, frozen = build(cfg)
    raw = raw.to(device, memory_format=torch.channels_last)
    if frozen is not None:
        frozen = frozen.to(device, memory_format=torch.channels_last)
    opt = torch.optim.Adam([p for p in raw.parameters() if p.requires_grad], lr=t["lr"])
    ema = copy.deepcopy(raw).eval().requires_grad_(False)
    step = 0
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optim"])
        step = ck["step"]
        if main0:
            print(f"resumed from {last} at step {step}", flush=True)
    model = DDP(raw, device_ids=[device.index]) if world > 1 else raw

    def save(path):
        if main0:
            torch.save({"step": step, "cfg": cfg, "model": raw.state_dict(), "ema": ema.state_dict(),
                        "optim": opt.state_dict()}, path + ".tmp")
            os.replace(path + ".tmp", path)

    restart_every = cfg["model"].get("vq_restart_every", 0) if cfg["stage"] == 2 else 0
    counts, window, t0 = None, {}, time.time()
    log = open(os.path.join(out_dir, "metrics.jsonl"), "a") if main0 else None
    epoch, batches = 0, None
    model.train()
    while step < t["max_steps"]:
        if batches is None:
            if sampler is not None:
                sampler.set_epoch(epoch)
            batches, epoch = iter(loader), epoch + 1
        batch = next(batches, None)
        if batch is None:
            batches = None
            continue
        inputs, targets = prepare_batch(batch, device, torch.bfloat16)
        with amp():
            loss, logs, out = compute_loss(cfg, model, frozen, inputs, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if t.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(raw.parameters(), t["grad_clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, t)
        opt.step()
        step += 1
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(raw.parameters()), 1 - t["ema"])
            for e, r in zip(ema.buffers(), raw.buffers()):  # BatchNorm statistics averaged like the weights
                if e.is_floating_point():
                    e.lerp_(r, 1 - t["ema"])
                else:
                    e.copy_(r)
        for k, v in logs.items():
            window[k] = window.get(k, 0.0) + v

        if restart_every:
            c = torch.stack([torch.stack([torch.bincount(i.flatten(), minlength=cfg["model"]["num_codes"])
                                          for i in out["idx"][m].unbind(1)]) for m in MODALITIES]).sum(0)
            counts = c if counts is None else counts + c
            if step % restart_every == 0:
                counts = all_reduce_sum(counts)
                if step <= cfg["model"]["vq_restart_until"]:
                    for lvl, (book, ema_book) in enumerate(zip(raw.vq.levels, ema.vq.levels)):
                        cand = torch.cat([r[lvl].float().permute(0, 2, 3, 1).reshape(-1, r[lvl].shape[1])
                                          for r in out["res"].values()])
                        dead = book.restart_dead_codes(counts[lvl], cand)
                        ema_book.codebook.weight.data[dead] = book.codebook.weight.data[dead]
                        state = opt.state.get(book.codebook.weight, {})
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in state:
                                state[key][dead] = 0
                counts = None
        del out, loss

        if step % t["log_every"] == 0:
            keys = sorted(window)
            vals = all_reduce_mean(torch.stack([window[k] / t["log_every"] for k in keys]).float())
            rec = {"step": step, **{k: float(v) for k, v in zip(keys, vals)}, "lr": opt.param_groups[0]["lr"],
                   "sec_per_step": (time.time() - t0) / t["log_every"]}
            if main0:
                print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            window, t0 = {}, time.time()
        if val_loader is not None and step % t["val_every"] == 0:
            rec = {"step": step, **validate(cfg, ema, frozen, val_loader, device, amp)}
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % t["ckpt_every"] == 0:
            save(last)
        if step % t["keep_every"] == 0:
            save(os.path.join(out_dir, "checkpoints", f"step_{step:07d}.pt"))
    save(last)
    cleanup()


if __name__ == "__main__":
    main()
