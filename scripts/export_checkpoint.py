"""Merges the LoRA of a train_sr.py checkpoint into the base VOSR model and writes a VOSR-format checkpoint folder
(args.json + ema_model.safetensors), usable by scripts/inference.py and by VOSR's own inference scripts.

    python scripts/export_checkpoint.py --base weights/VOSR/VOSR_1.4B_os --lora runs/sr/checkpoints/last.pt --out MURL-SR_1.4B_os
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.vosr import export  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="VOSR-format folder of the model the LoRA was trained on")
    ap.add_argument("--lora", required=True, help="checkpoint of scripts/train_sr.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.lora, map_location="cpu", weights_only=False)
    lc = ck["cfg"]["train"]["lora"]
    export(args.base, ck["lora"], lc["rank"], lc["alpha"], args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
