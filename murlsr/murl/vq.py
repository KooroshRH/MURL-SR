import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Nearest-codeword quantization with the VQ-VAE loss ||sg[u] - q||^2 + beta ||u - sg[q]||^2 and the
    straight-through estimator, in fp32."""

    def __init__(self, num_codes=8192, dim=128, beta=1.0, chunk=32768):
        super().__init__()
        self.beta, self.chunk = beta, chunk
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(self, z):
        B, C, H, W = z.shape
        with torch.autocast(z.device.type, enabled=False):
            zf = z.float().permute(0, 2, 3, 1).reshape(-1, C)
            w = self.codebook.weight.float()
            with torch.no_grad():
                # exact fp32 search (TF32 off), in row chunks to bound the N x K buffer
                tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    w2 = w.pow(2).sum(1)
                    idx = torch.cat([torch.addmm(w2, c, w.t(), alpha=-2).argmin(1) for c in zf.split(self.chunk)])
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
            q = F.embedding(idx, w)
            loss = F.mse_loss(q, zf.detach()) + self.beta * F.mse_loss(zf, q.detach())
            q = zf + (q - zf).detach()
        return q.view(B, H, W, C).permute(0, 3, 1, 2), loss, idx.view(B, H, W)

    @torch.no_grad()
    def restart_dead_codes(self, counts, candidates):
        """Re-seeds unused codes with random rows of `candidates` (N, dim); rank 0's codebook is broadcast under DDP."""
        dead = (counts == 0).nonzero().flatten()
        if dead.numel():
            pick = torch.randint(0, candidates.shape[0], (dead.numel(),), device=candidates.device)
            self.codebook.weight.data[dead] = candidates[pick].float()
        if dist.is_initialized():
            dist.broadcast(self.codebook.weight.data, 0)
        return dead


class ResidualVQ(nn.Module):
    """Residual quantization with the shared codebooks: level l quantizes what the levels before it left."""

    def __init__(self, levels=4, num_codes=8192, dim=128, beta=1.0):
        super().__init__()
        self.levels = nn.ModuleList(VectorQuantizer(num_codes, dim, beta) for _ in range(levels))

    def forward(self, z):
        """q, VQ loss, code indices (B, levels, H, W) and the input of each level (code restart candidates)."""
        r, total, loss, idx, res = z, 0, 0, [], []
        for vq in self.levels:
            res.append(r.detach())
            q, l, i = vq(r)
            q = q.detach()
            r = r - q
            total, loss = total + q, loss + l
            idx.append(i)
        return z + (total - z).detach(), loss, torch.stack(idx, 1), res
