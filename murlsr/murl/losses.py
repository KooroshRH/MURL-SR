import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-6):
    """Multi-class soft Dice loss, mean over batch and classes."""
    p = torch.softmax(logits.float(), dim=1)
    B, C = p.shape[:2]
    t = target.flatten(1)
    inter = p.new_zeros(B, C).scatter_add_(1, t, p.gather(1, target.unsqueeze(1)).flatten(1))
    count = p.new_zeros(B, C).scatter_add_(1, t, p.new_ones(t.shape))
    return 1 - ((2 * inter + eps) / (p.sum((2, 3)) + count + eps)).mean()


def reconstruction_loss(modality, pred, target):
    """MSE for the visual and depth branches, cross-entropy + Dice for segmentation."""
    if modality == "seg":
        return 0.5 * F.cross_entropy(pred.float(), target) + 0.5 * dice_loss(pred, target)
    return F.mse_loss(pred.float(), target)


def orthogonality_loss(spec, uni):
    """Eq. 10: ||E_uni(I)^T E_spec(I)||_F^2 with spatial locations as rows, on features scaled to unit Frobenius norm
    per image so the loss lies in [0, 1]. Mean over the batch."""
    with torch.autocast(uni.device.type, enabled=False):
        s = F.normalize(spec.float().flatten(2), dim=(1, 2))
        u = F.normalize(uni.float().flatten(2), dim=(1, 2))
        return (u @ s.transpose(1, 2)).pow(2).sum((1, 2)).mean()
