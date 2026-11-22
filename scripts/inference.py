"""Super-resolve images with MURL-SR, or with any VOSR-1.4B one-step checkpoint in VOSR's format.

    python scripts/inference.py -i path/to/lr_images -o results
    python scripts/inference.py -i lr.png -o results --checkpoint path/to/VOSR_1.4B_os
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from murlsr.data import EXTENSIONS, to_tensor, to_uint8  # noqa: E402
from murlsr.vosr import VOSR, wavelet_color_fix  # noqa: E402

MODEL_REPO = "kooroshrh/MURL-SR"
VOSR_REPO = "CSWRY/VOSR"


def download(repo, patterns):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, allow_patterns=patterns)


@torch.no_grad()
def super_resolve(model, lr, upscale, seed, color_fix=True):
    """LR PIL image -> SR PIL image (upscale x larger)."""
    lq = lr.resize((lr.width * upscale, lr.height * upscale), Image.BICUBIC)
    x = to_tensor(lq, model.lat_mean.device) * 2 - 1
    H, W = x.shape[-2:]
    ph, pw = -H % 16, -W % 16  # latent sides must be even (VAE x8, DiT patch 2)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    cond = model.condition(x)
    g = torch.Generator(x.device).manual_seed(seed)
    z = model.sample(cond, torch.randn(cond["lq"].shape, generator=g, device=x.device))
    sr = Image.fromarray(to_uint8(model.decode(z)[..., :H, :W] * 0.5 + 0.5)[0])
    return wavelet_color_fix(sr, lq) if color_fix else sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="an image or a folder of images")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--checkpoint", default=None, help=f"VOSR-format checkpoint folder (default: {MODEL_REPO} from Hugging Face)")
    ap.add_argument("--vae", default=None, help=f"Qwen-Image-vae-2d folder (default: from {VOSR_REPO} on Hugging Face)")
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-color-fix", action="store_true", help="skip VOSR's wavelet color correction")
    args = ap.parse_args()

    ckpt = args.checkpoint or download(MODEL_REPO, ["args.json", "ema_model.safetensors"])
    vae = args.vae or os.path.join(download(VOSR_REPO, ["Qwen-Image-vae-2d/*"]), "Qwen-Image-vae-2d")
    model = VOSR(ckpt, vae, "cuda")
    paths = [args.input] if os.path.isfile(args.input) else sorted(
        os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(EXTENSIONS))
    os.makedirs(args.output, exist_ok=True)
    for p in paths:
        sr = super_resolve(model, Image.open(p).convert("RGB"), args.upscale, args.seed, not args.no_color_fix)
        sr.save(os.path.join(args.output, os.path.splitext(os.path.basename(p))[0] + ".png"))
        print(os.path.basename(p), flush=True)


if __name__ == "__main__":
    main()
