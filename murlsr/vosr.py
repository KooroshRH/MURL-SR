"""VOSR-1.4B one-step (https://github.com/cswry/VOSR) as the SR backbone. A checkpoint is a folder in VOSR's own
format, `args.json` plus `ema_model.safetensors` (directly or under `checkpoints/`), so MURL-SR and VOSR checkpoints
are interchangeable here and in VOSR's inference script. Flow convention: z_t = (1 - t) z_0 + t eps; the DiT predicts
the mean velocity u(z_t, t, r) and one-step sampling is z_0 = eps - u(eps, 1, 0). Pixels are in [-1, 1]; the LQ input
is the LR image bicubic-upsampled to the output size."""
import contextlib
import json
import os
import types

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

LORA_TARGETS = r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(w12|w3)|cross_attn\.(q_linear|k_linear|v_linear|proj))"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def weights_file(checkpoint):
    for d in (os.path.join(checkpoint, "checkpoints"), checkpoint):
        for name in ("ema_model.safetensors", "model.safetensors"):
            if os.path.isfile(os.path.join(d, name)):
                return os.path.join(d, name)
    raise FileNotFoundError(f"no ema_model.safetensors under {checkpoint}")


def _disable_compile_without_triton():
    # the DiT decorates helpers with torch.compile, which needs triton
    try:
        import triton  # noqa: F401
    except ImportError:
        import torch._dynamo

        torch._dynamo.config.disable = True


def load_dit(checkpoint):
    """The DiT of a VOSR-format checkpoint folder and its args.json (one-step, Qwen VAE models)."""
    _disable_compile_without_triton()
    from safetensors.torch import load_file

    from .third_party.vosr.lightningdit import LightningDiT

    with open(os.path.join(checkpoint, "args.json")) as f:
        a = types.SimpleNamespace(**json.load(f))
    assert hasattr(a, "distill_type") and a.ae_type == "qwen", f"{checkpoint} is not a VOSR one-step (Qwen VAE) model"
    aux = getattr(a, "auxiliary_time_cond", None)
    dit = LightningDiT(input_size=a.resolution // 8, patch_size=a.patch_size, in_channels=32, out_channels=16,
                       hidden_size=a.dim, depth=a.depth, num_heads=a.num_heads, mlp_ratio=a.mlp_ratio, z_dims=a.enc_dim,
                       encdim_ratio=a.encdim_ratio, auxiliary_time_cond=True if aux is None else bool(aux),
                       use_qknorm=a.use_qknorm, use_swiglu=a.use_swiglu, use_rope=a.use_rope, use_rmsnorm=a.use_rmsnorm,
                       wo_shift=a.wo_shift, num_fused_layers=len(a.layer_dinov2b_list))
    missing, unexpected = dit.load_state_dict(load_file(weights_file(checkpoint)), strict=False)
    assert not unexpected and all("rope" in k for k in missing), (missing, unexpected)
    return dit, a


def add_lora(dit, rank, alpha):
    """Freezes the DiT and adds LoRA (output unchanged at init) to the Linear layers of every block."""
    from peft import LoraConfig, inject_adapter_in_model

    dit.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, target_modules=LORA_TARGETS), dit)
    params = [p for n, p in dit.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad_(True)
    return params


def lora_state_dict(dit):
    return {n: p.detach().cpu() for n, p in dit.named_parameters() if "lora_" in n}


def load_lora(dit, sd):
    missing = {n for n, _ in dit.named_parameters() if "lora_" in n} - set(sd)
    assert not missing, f"LoRA weights missing: {sorted(missing)[:3]}"
    dit.load_state_dict(sd, strict=False)


def merge_lora(dit):
    """Folds the LoRA weights into the base Linear layers and removes the adapters."""
    names = {id(m): n for n, m in dit.named_modules()}
    for m in _lora_layers(dit):
        m.merge()
        parent, _, child = names[id(m)].rpartition(".")
        setattr(dit.get_submodule(parent) if parent else dit, child, m.get_base_layer())
    if hasattr(dit, "peft_config"):
        del dit.peft_config


def _lora_layers(module):
    from peft.tuners.tuners_utils import BaseTunerLayer

    return [m for m in module.modules() if isinstance(m, BaseTunerLayer)]


class VOSR(nn.Module):
    def __init__(self, checkpoint, vae, device="cuda"):
        """checkpoint: VOSR-format folder of the DiT; vae: folder of VOSR's Qwen-Image 2D VAE (Qwen-Image-vae-2d)."""
        super().__init__()
        from .third_party.vosr.qwenimage_vae2d import AutoencoderKLQwenImage2D

        self.dit, a = load_dit(checkpoint)
        self.args = a
        self.vae = AutoencoderKLQwenImage2D.from_pretrained(vae)
        self.register_buffer("lat_mean", torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("lat_istd", 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1), persistent=False)
        self.venc = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{a.enc_type[len('dinov2'):]}14", trust_repo=True)
        self.venc.head = nn.Identity()
        self.register_buffer("im_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.requires_grad_(False).eval().to(device)

    def encode(self, x):
        """[-1, 1] pixels -> normalized latent (posterior mode)."""
        return (self.vae.encode(x).latent_dist.mode() - self.lat_mean) * self.lat_istd

    def decode(self, z):
        """Normalized latent -> [-1, 1] pixels."""
        return self.vae.decode(z / self.lat_istd + self.lat_mean, return_dict=False)[0].clamp(-1, 1)

    @torch.no_grad()
    def condition(self, lq):
        """LQ in [-1, 1] at the output size -> its latent and DINOv2 features (VOSR's get_venc_features arithmetic)."""
        x = ((0.5 * lq + 0.5) * 255) / 255.0
        x = torch.nn.functional.interpolate(x, self.args.dinov2_size, mode="bicubic").clip(0.0, 1.0)
        v = self.venc.prepare_tokens_with_masks((x - self.im_mean) / self.im_std)
        last, want, layers = len(self.venc.blocks) - 1, self.args.layer_dinov2b_list, {}
        for i, blk in enumerate(self.venc.blocks[: max(want) + 1]):
            v = blk(v)
            layers[i] = (self.venc.norm(v) if i == last else v)[:, 1:]
        return {"lq": self.encode(lq), "venc": [layers[i] for i in want]}

    def velocity(self, z, t, cond, r):
        """Mean velocity u(z_t, t, r)."""
        return self.dit.forward_flexible(torch.cat([cond["lq"], z], 1), t, r, z=cond["venc"])

    @torch.no_grad()
    def sample(self, cond, noise):
        """One-step sample from `noise` (one per image): z_0 = eps - u(eps, 1, 0)."""
        n = noise.shape[0]
        return noise - self.velocity(noise, noise.new_ones(n), cond, noise.new_zeros(n))

    def add_lora(self, rank, alpha):
        return add_lora(self.dit, rank, alpha)

    def lora_state_dict(self):
        return lora_state_dict(self.dit)

    @contextlib.contextmanager
    def reference(self):
        """The frozen pretrained DiT (LoRA off, no gradient checkpointing): the DPO reference model."""
        layers, ckpt = _lora_layers(self.dit), self.dit.use_checkpoint
        for m in layers:
            m.enable_adapters(False)
        self.dit.use_checkpoint = False
        try:
            yield
        finally:
            self.dit.use_checkpoint = ckpt
            for m in layers:
                m.enable_adapters(True)

    def gradient_checkpointing(self):
        # VOSR checkpoints its blocks with use_reentrant=True, which gives no gradient when no block input requires
        # grad (true with LoRA); marking the patch embedding output as requiring grad fixes it.
        self.dit.use_checkpoint = True
        self.dit.x_embedder.register_forward_hook(lambda m, i, o: o.requires_grad_(True))


def wavelet_color_fix(sr, lq):
    """VOSR's default post-processing: low frequencies of the LQ, high frequencies of the SR (PIL images)."""
    import cv2

    t = np.asarray(sr).astype(np.float32) / 255.0
    s = np.asarray(lq.resize(sr.size, Image.LANCZOS)).astype(np.float32) / 255.0
    out = cv2.GaussianBlur(s, (0, 0), 5) + (t - cv2.GaussianBlur(t, (0, 0), 5))
    return Image.fromarray((np.clip(out, 0, 1) * 255.0).astype(np.uint8))


def export(base, lora, rank, alpha, out):
    """Writes a VOSR-format checkpoint folder `out`: the DiT of `base` with the LoRA weights `lora` merged in, with
    the same tensor names and dtypes as the base file, and the base args.json."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    dit, _ = load_dit(base)
    add_lora(dit, rank, alpha)
    load_lora(dit, lora)
    merge_lora(dit)
    sd = dit.state_dict()
    dtypes = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    with safe_open(weights_file(base), "pt") as f:
        merged = {k: sd[k].to(dtypes[f.get_slice(k).get_dtype()]).contiguous() for k in f.keys()}
    os.makedirs(out, exist_ok=True)
    save_file(merged, os.path.join(out, "ema_model.safetensors"))
    shutil.copy(os.path.join(base, "args.json"), os.path.join(out, "args.json"))
