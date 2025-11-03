"""
Denoising metrics: Comprehensive evaluation metrics for pose denoising.
"""
import numpy as np
import torch
from typing import Dict


def compute_rmse(pred, target, mask=None):
    """
    Root Mean Squared Error.
    
    Args:
        pred: (B, 3, 13) or (B, 39) predicted poses
        target: (B, 3, 13) or (B, 39) target poses
        mask: (B, 13) or (B, 39) optional mask (1=include, 0=exclude)
    """
    if pred.dim() == 3:  # (B, 3, 13)
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    
    se = (pred - target) ** 2  # (B, 39)
    
    if mask is not None:
        if mask.dim() == 2 and mask.shape[1] == 13:  # (B, 13) -> expand to (B, 39)
            mask = mask.repeat_interleave(3, dim=1)
        se = se * mask
        count = mask.sum().clamp(min=1.0)
        return torch.sqrt(se.sum() / count).item()
    else:
        return torch.sqrt(se.mean()).item()


def compute_mae(pred, target, mask=None):
    """Mean Absolute Error."""
    if pred.dim() == 3:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    
    ae = torch.abs(pred - target)
    
    if mask is not None:
        if mask.dim() == 2 and mask.shape[1] == 13:
            mask = mask.repeat_interleave(3, dim=1)
        ae = ae * mask
        count = mask.sum().clamp(min=1.0)
        return (ae.sum() / count).item()
    else:
        return ae.mean().item()


def compute_per_keypoint_rmse(pred, target, mask=None):
    """
    Per-keypoint RMSE.
    
    Returns:
        (13,) array of RMSE per keypoint
    """
    if pred.dim() == 2:  # (B, 39) -> (B, 3, 13)
        pred = pred.reshape(pred.shape[0], 3, 13)
        target = target.reshape(target.shape[0], 3, 13)
    
    # (B, 3, 13)
    se = (pred - target) ** 2
    
    if mask is not None:
        if mask.dim() == 2 and mask.shape[1] == 13:  # (B, 13)
            mask = mask.unsqueeze(1)  # (B, 1, 13)
        se = se * mask
        count = mask.sum(dim=0).clamp(min=1.0)  # (1, 13)
        mse = se.sum(dim=0).sum(dim=0) / count.squeeze()  # (13,)
    else:
        mse = se.mean(dim=0).mean(dim=0)  # (13,)
    
    return torch.sqrt(mse).cpu().numpy()


def compute_temporal_smoothness(poses):
    """
    Measure temporal smoothness (frame-to-frame displacement).
    Lower is smoother.
    
    Args:
        poses: (T, 3, 13) or (T, 39) sequence of poses
    """
    if poses.dim() == 2:  # (T, 39)
        poses = poses.reshape(poses.shape[0], 3, 13)
    
    # Compute displacements
    displacements = torch.norm(poses[1:] - poses[:-1], dim=1)  # (T-1, 13)
    return displacements.mean(dim=0).cpu().numpy()  # (13,)


def compute_anatomical_consistency(poses):
    """
    Measure how well poses maintain anatomical constraints.
    Uses inter-keypoint distances as constraints.
    
    Args:
        poses: (B, 3, 13) batch of poses
    """
    if poses.dim() == 2:
        poses = poses.reshape(poses.shape[0], 3, 13)
    
    # Compute inter-keypoint distances for each pose
    # This is a simplified version - could use skeleton constraints
    kp = poses.transpose(1, 2)  # (B, 13, 3)
    
    # Compute all pairwise distances
    distances = []
    for b in range(poses.shape[0]):
        dists = torch.cdist(kp[b:b+1], kp[b:b+1]).squeeze()  # (13, 13)
        distances.append(dists)
    
    distances = torch.stack(distances)  # (B, 13, 13)
    
    # Measure variance across batch (should be low for consistent anatomy)
    dist_std = distances.std(dim=0).mean().item()
    
    return dist_std


def compute_all_metrics(pred, target, mask=None, pred_sequence=None, target_sequence=None):
    """
    Compute comprehensive metrics.
    
    Args:
        pred: (B, 3, 13) or (B, 39) predicted poses
        target: (B, 3, 13) or (B, 39) target poses
        mask: (B, 13) or (B, 39) optional mask
        pred_sequence: (T, 3, 13) optional sequence for temporal metrics
        target_sequence: (T, 3, 13) optional sequence for temporal metrics
    """
    metrics = {}
    
    # Spatial accuracy
    metrics['rmse'] = compute_rmse(pred, target, mask)
    metrics['mae'] = compute_mae(pred, target, mask)
    metrics['per_keypoint_rmse'] = compute_per_keypoint_rmse(pred, target, mask)
    
    # Masked vs unmasked breakdown
    if mask is not None:
        if mask.dim() == 2 and mask.shape[1] == 13:
            mask_flat = mask.repeat_interleave(3, dim=1) if pred.dim() == 2 else mask
        else:
            mask_flat = mask
            
        masked_pred = pred if pred.dim() == 2 else pred.reshape(pred.shape[0], -1)
        masked_target = target if target.dim() == 2 else target.reshape(target.shape[0], -1)
        
        masked_only = mask_flat < 0.5  # masked regions
        unmasked_only = mask_flat > 0.5  # unmasked regions
        
        if masked_only.sum() > 0:
            metrics['rmse_masked'] = compute_rmse(pred, target, masked_only.float())
        if unmasked_only.sum() > 0:
            metrics['rmse_unmasked'] = compute_rmse(pred, target, unmasked_only.float())
    
    # Temporal smoothness
    if pred_sequence is not None:
        metrics['temporal_smoothness_pred'] = compute_temporal_smoothness(pred_sequence).mean()
        if target_sequence is not None:
            metrics['temporal_smoothness_target'] = compute_temporal_smoothness(target_sequence).mean()
            metrics['smoothness_ratio'] = (
                metrics['temporal_smoothness_pred'] / 
                metrics['temporal_smoothness_target']
            )
    
    # Anatomical consistency
    if pred.dim() == 3:
        metrics['anatomical_consistency'] = compute_anatomical_consistency(pred)
    
    return metrics



