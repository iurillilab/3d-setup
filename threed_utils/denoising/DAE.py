import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_pose(x):  # (B,3,13,1)->(B,39)
    return x.view(x.size(0), -1)


def build_mask(x, fill=0.0):  # 1=observed, 0=masked
    return (x != fill).float().view(x.size(0), -1)


class DAE(nn.Module):
    def __init__(self, d=39, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, h),
            nn.GELU(),
            nn.Linear(h, d),
        )

    def forward(self, x_flat, m_flat):
        # predict residual; blend to copy unmasked straight through
        residual = self.net(torch.cat([x_flat, m_flat], dim=-1))
        recon = m_flat * x_flat + (1 - m_flat) * (x_flat + residual)
        return recon, residual


def loss_masked(pred, target, m_flat, lam: float = 0.05):
    """
    pred,target: (B,39)
    m_flat:      (B,39)  1=observed, 0=masked
    """
    se = (pred - target) ** 2
    masked = 1 - m_flat
    unmasked = m_flat

    # normalize per sample by count of masked/unmasked, then average batch
    masked_cnt = masked.sum(dim=1).clamp(min=1.0)
    unmasked_cnt = unmasked.sum(dim=1).clamp(min=1.0)

    masked_loss = ((se * masked).sum(dim=1) / masked_cnt).mean()
    unmasked_loss = ((se * unmasked).sum(dim=1) / unmasked_cnt).mean()

    return masked_loss + lam * unmasked_loss
