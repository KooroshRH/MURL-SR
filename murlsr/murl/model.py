import torch
import torch.nn as nn

from .attention import CrossAttention, Fusion
from .decoders import ResDecoder
from .encoders import LATENT_CHANNELS, ResEncoder, ResNetEncoder
from .vq import ResidualVQ

MODALITIES = ("vis", "seg", "depth")
NUM_SEG_CLASSES = 150
IN_CHANNELS = {"vis": 3, "seg": NUM_SEG_CLASSES, "depth": 1}


def make_encoder(m, cfg, pretrained=True):
    c, stride = cfg[m], cfg["latent_stride"]
    if m == "vis":
        return ResEncoder(3, c["enc_channels"], c["enc_blocks"], stride, c.get("grad_checkpoint", False))
    return ResNetEncoder(IN_CHANNELS[m], stride, pretrained)


def make_decoder(m, cfg):
    c = cfg[m]
    return ResDecoder(IN_CHANNELS[m], c["dec_channels"], c["dec_blocks"], cfg["latent_stride"], sigmoid=m != "seg",
                      grad_checkpoint=c.get("grad_checkpoint", False))


def prepare_batch(batch, device, onehot_dtype=torch.float32):
    """Loader batch (rgb uint8, seg uint8 labels, depth float) -> encoder inputs and reconstruction targets."""
    rgb, seg, depth = (t.to(device, non_blocking=True) for t in batch)
    vis = (rgb.float() / 255).contiguous(memory_format=torch.channels_last)
    seg = seg.long()
    depth = depth.contiguous(memory_format=torch.channels_last)
    inputs = {"vis": vis, "seg": seg_onehot(seg, onehot_dtype), "depth": depth}
    return inputs, {"vis": vis, "seg": seg, "depth": depth}


def seg_onehot(seg, dtype=torch.float32):
    B, H, W = seg.shape
    onehot = torch.zeros(B, NUM_SEG_CLASSES, H, W, dtype=dtype, device=seg.device).to(memory_format=torch.channels_last)
    return onehot.scatter_(1, seg.unsqueeze(1), 1)


class ModalityAE(nn.Module):
    def __init__(self, m, cfg, pretrained=True):
        super().__init__()
        self.encoder = make_encoder(m, cfg, pretrained)
        self.decoder = make_decoder(m, cfg)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class SpecificAEs(nn.Module):
    """Stage 1: one autoencoder per modality, trained jointly with independent losses."""

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.aes = nn.ModuleDict({m: ModalityAE(m, cfg, pretrained) for m in MODALITIES})

    def forward(self, inputs):
        out = {m: self.aes[m](inputs[m]) for m in MODALITIES}
        return {"recon": {m: o[0] for m, o in out.items()}, "z": {m: o[1] for m, o in out.items()}}


class MURL(nn.Module):
    """Stage 2: unified encoders, cross-attention of each modality (query) to the fused other two (keys and values),
    a shared residual codebook and unified decoders."""

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        M = MODALITIES
        self.enc = nn.ModuleDict({m: make_encoder(m, cfg, pretrained) for m in M})
        self.fuse = nn.ModuleDict({m: Fusion((len(M) - 1) * LATENT_CHANNELS, LATENT_CHANNELS) for m in M})
        kv_pool = cfg.get("ca_kv_stride", cfg["latent_stride"]) // cfg["latent_stride"]
        self.ca = nn.ModuleDict({m: CrossAttention(LATENT_CHANNELS, cfg["heads"], kv_pool) for m in M})
        self.vq = ResidualVQ(cfg["vq_levels"], cfg["num_codes"], LATENT_CHANNELS, cfg["vq_beta"])
        self.dec = nn.ModuleDict({m: make_decoder(m, cfg) for m in M})

    def encode(self, inputs):
        """Unified encoder outputs z and continuous unified representations u (after cross-attention)."""
        z = {m: self.enc[m](inputs[m]) for m in MODALITIES}
        u = {m: self.ca[m](z[m], self.fuse[m](*[z[o] for o in MODALITIES if o != m])) for m in MODALITIES}
        return z, u

    def features(self, inputs):
        """u (pre-quantization) and q (post-quantization) per modality, without decoding."""
        _, u = self.encode(inputs)
        return u, {m: self.vq(u[m])[0] for m in MODALITIES}

    def forward(self, inputs):
        z, u = self.encode(inputs)
        q, vq_loss, idx, res, recon = {}, {}, {}, {}, {}
        for m in MODALITIES:
            q[m], vq_loss[m], idx[m], res[m] = self.vq(u[m])
            recon[m] = self.dec[m](q[m])
        return {"z": z, "u": u, "q": q, "vq_loss": vq_loss, "idx": idx, "res": res, "recon": recon}
