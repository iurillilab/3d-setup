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


def loss_masked(pred, target, m_flat, lam=0.05):
    # main: only masked entries
    masked = 1 - m_flat
    masked_mse = F.mse_loss(
        pred * masked, target * masked, reduction="sum"
    ) / masked.sum().clamp(min=1.0)
    # tiny consistency on unmasked to keep pose coherent
    unmasked = m_flat
    unmasked_mse = F.mse_loss(
        pred * unmasked, target * unmasked, reduction="sum"
    ) / unmasked.sum().clamp(min=1.0)
    return masked_mse + lam * unmasked_mse
