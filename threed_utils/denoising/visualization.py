"""
Visualization utilities for denoising model training.
Contains plotting functions for 3D poses, keypoint errors, and training progress.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_equal_3d(ax):
    """Set equal aspect ratio for 3D plots."""
    xs, ys, zs = [getattr(ax, f"get_{a}lim")() for a in "xyz"]
    ranges = np.array([xs, ys, zs])
    minv, maxv = ranges[:, 0].min(), ranges[:, 1].max()
    center = (minv + maxv) / 2.0
    span = (maxv - minv)
    half = span / 2.0
    ax.set_xlim(center - half, center + half)
    ax.set_ylim(center - half, center + half)
    ax.set_zlim(center - half, center + half)


def plot_pose3d_grid(masked, recon, target, obs_mask, edges=None, keypoint_names=None, title="Pose3D"):
    """
    Plot 3D poses with observed/masked markers and reconstruction.
    
    Args:
        masked/recon/target: (B, 3, 13) tensors
        obs_mask: (B, 13) tensor with 1=observed, 0=masked
        edges: list of (i,j) pairs for skeleton connections
        keypoint_names: list of keypoint names for labeling
        title: plot title
    """
    B = masked.shape[0]
    cols, rows = 2, int(np.ceil(B / 2))
    fig = plt.figure(figsize=(cols * 6, rows * 6))
    
    for i in range(B):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        m = masked[i].cpu().numpy() if hasattr(masked[i], 'cpu') else masked[i]
        r = recon[i].cpu().numpy() if hasattr(recon[i], 'cpu') else recon[i]
        t = target[i].cpu().numpy() if hasattr(target[i], 'cpu') else target[i]
        mask = obs_mask[i].cpu().numpy().astype(bool) if hasattr(obs_mask[i], 'cpu') else obs_mask[i].astype(bool)

        # observed vs masked inputs
        ax.scatter(m[0, mask], m[1, mask], m[2, mask], s=30, marker="o", c="blue",
                   label="Input observed", alpha=0.8)
        ax.scatter(m[0, ~mask], m[1, ~mask], m[2, ~mask], s=50, marker="x", c="orange",
                   label="Input masked", alpha=0.8)

        # recon & target
        ax.scatter(r[0], r[1], r[2], s=25, marker=".", c="green", label="Reconstruction", alpha=0.8)
        ax.scatter(t[0], t[1], t[2], s=25, marker="^", c="red", label="Target", alpha=0.8)

        # recon→target arrows for masked points only
        for j in np.where(~mask)[0]:
            ax.plot([r[0, j], t[0, j]], [r[1, j], t[1, j]], [r[2, j], t[2, j]],
                   linestyle="--", linewidth=2, alpha=0.7, color="purple")

        # skeleton connections
        if edges is not None:
            for (a, b) in edges:
                ax.plot(t[0, [a, b]], t[1, [a, b]], t[2, [a, b]],
                       linewidth=2, alpha=0.4, color="gray")

        ax.set_title(f"Sample {i}", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_equal_3d(ax)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_pose2d_projections(masked, recon, target, obs_mask, edges=None, keypoint_names=None, title="Pose2D Projections"):
    """
    Create 2D projections (XY, XZ, YZ) for better visualization.
    
    Args:
        masked/recon/target: (B, 3, 13) tensors
        obs_mask: (B, 13) tensor with 1=observed, 0=masked
        edges: list of (i,j) pairs for skeleton connections
        keypoint_names: list of keypoint names for labeling
        title: plot title
    """
    B = masked.shape[0]
    projections = [('XY', [0, 1]), ('XZ', [0, 2]), ('YZ', [1, 2])]

    fig, axes = plt.subplots(B, 3, figsize=(15, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        m = masked[i].cpu().numpy() if hasattr(masked[i], 'cpu') else masked[i]
        r = recon[i].cpu().numpy() if hasattr(recon[i], 'cpu') else recon[i]
        t = target[i].cpu().numpy() if hasattr(target[i], 'cpu') else target[i]
        mask = obs_mask[i].cpu().numpy().astype(bool) if hasattr(obs_mask[i], 'cpu') else obs_mask[i].astype(bool)

        for j, (proj_name, (ax1, ax2)) in enumerate(projections):
            ax = axes[i, j]

            # observed vs masked inputs
            ax.scatter(m[ax1, mask], m[ax2, mask], s=30, marker="o", c="blue",
                      label="Input observed", alpha=0.8)
            ax.scatter(m[ax1, ~mask], m[ax2, ~mask], s=50, marker="x", c="orange",
                      label="Input masked", alpha=0.8)

            # recon & target
            ax.scatter(r[ax1], r[ax2], s=25, marker=".", c="green", label="Reconstruction", alpha=0.8)
            ax.scatter(t[ax1], t[ax2], s=25, marker="^", c="red", label="Target", alpha=0.8)

            # recon→target arrows for masked points only
            for k in np.where(~mask)[0]:
                ax.plot([r[ax1, k], t[ax1, k]], [r[ax2, k], t[ax2, k]],
                       linestyle="--", linewidth=2, alpha=0.7, color="purple")

            # skeleton connections
            if edges is not None:
                for (a, b) in edges:
                    ax.plot(t[ax1, [a, b]], t[ax2, [a, b]],
                           linewidth=2, alpha=0.4, color="gray")

            ax.set_title(f"Sample {i} - {proj_name}")
            ax.set_xlabel(f"{proj_name[0]}")
            ax.set_ylabel(f"{proj_name[1]}")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_keypoint_error_bar(err_xyz, keypoint_names=None, title="Keypoint RMSE (sorted)"):
    """
    Plot per-keypoint RMSE as a bar chart, sorted by error.
    
    Args:
        err_xyz: (13, 3) array of per-axis RMSE per keypoint
        keypoint_names: list of keypoint names
        title: plot title
    """
    rmse = np.linalg.norm(err_xyz, axis=1)  # (13,)
    order = np.argsort(-rmse)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars with colors based on error magnitude
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, 13))
    bars = ax.bar(np.arange(13), rmse[order], color=colors)

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, rmse[order])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Set labels
    if keypoint_names is not None:
        ax.set_xticks(np.arange(13))
        ax.set_xticklabels([keypoint_names[i] for i in order], rotation=45, ha='right')
    else:
        ax.set_xticks(np.arange(13))
        ax.set_xticklabels([f'KP {i}' for i in order])

    ax.set_ylabel("RMSE")
    ax.set_xlabel("Keypoint (sorted by error)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_keypoint_error_progression(error_history, keypoint_names=None, title="Keypoint Error Progression"):
    """
    Plot the progression of keypoint errors over training steps.
    
    Args:
        error_history: list of arrays, each of shape (13,) - RMSE per keypoint per step
        keypoint_names: list of keypoint names
        title: plot title
    """
    if len(error_history) < 2:
        return None

    error_history = np.array(error_history)  # (steps, 13)
    steps = np.arange(len(error_history))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Individual keypoint error progression
    colors = plt.cm.tab20(np.linspace(0, 1, 13))
    for i in range(13):
        label = keypoint_names[i] if keypoint_names else f'KP {i}'
        ax1.plot(steps, error_history[:, i], color=colors[i], label=label, linewidth=2)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Individual Keypoint Error Progression")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 2: Overall error statistics
    mean_errors = np.mean(error_history, axis=1)
    std_errors = np.std(error_history, axis=1)
    max_errors = np.max(error_history, axis=1)
    min_errors = np.min(error_history, axis=1)

    ax2.plot(steps, mean_errors, 'b-', label='Mean', linewidth=2)
    ax2.fill_between(steps, mean_errors - std_errors, mean_errors + std_errors,
                    alpha=0.3, color='blue', label='±1 std')
    ax2.plot(steps, max_errors, 'r--', label='Max', linewidth=1)
    ax2.plot(steps, min_errors, 'g--', label='Min', linewidth=1)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Overall Error Statistics")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_mask_error_summary(mask_ratios, masked_errors, unmasked_errors, steps=None, title="Mask Ratio & Error Summary"):
    """
    Create clear summary plot for mask ratio and masked/unmasked errors.
    
    Args:
        mask_ratios: list of mask ratios over time
        masked_errors: list of masked region errors
        unmasked_errors: list of unmasked region errors  
        steps: optional step indices (defaults to range(len(mask_ratios)))
        title: plot title
    """
    if steps is None:
        steps = np.arange(len(mask_ratios))
    
    # Larger figure for better visibility in TensorBoard
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=100)
    
    # Plot 1: Mask Ratio over time
    ax1.plot(steps, mask_ratios, 'o-', color='#6A1B9A', linewidth=3, markersize=6, 
             label='Mask Ratio', alpha=0.8, markerfacecolor='white', markeredgewidth=2)
    mean_mask = np.mean(mask_ratios)
    ax1.axhline(mean_mask, color='#C62828', linestyle='--', linewidth=2.5, 
                label=f'Mean: {mean_mask:.3f}', alpha=0.7)
    ax1.axhline(np.mean(mask_ratios) + np.std(mask_ratios), color='#C62828', 
                linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axhline(np.mean(mask_ratios) - np.std(mask_ratios), color='#C62828', 
                linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.fill_between(steps, 0, mask_ratios, alpha=0.15, color='purple')
    ax1.set_ylabel('Mask Ratio\n(Fraction of keypoints masked)', fontsize=13, fontweight='bold')
    ax1.set_title('Mask Ratio Over Training', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, max(mask_ratios) * 1.25])
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_xticklabels([])
    ax1.tick_params(axis='y', labelsize=11)
    
    # Add text box with statistics (larger, clearer)
    stats_text = f'Statistics:\nMin: {np.min(mask_ratios):.3f}\nMax: {np.max(mask_ratios):.3f}\nMean: {np.mean(mask_ratios):.3f}\nStd: {np.std(mask_ratios):.3f}'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5),
             fontsize=10, family='monospace', fontweight='bold')
    
    # Plot 2: Masked vs Unmasked Errors
    ax2.plot(steps, masked_errors, 'o-', color='#FF6F00', linewidth=3, markersize=6,
             label='Masked Keypoints Error', alpha=0.85, markerfacecolor='white', markeredgewidth=2)
    ax2.plot(steps, unmasked_errors, 's-', color='#2E7D32', linewidth=3, markersize=6,
             label='Unmasked Keypoints Error', alpha=0.85, markerfacecolor='white', markeredgewidth=2)
    
    # Add mean lines with std bands
    masked_mean = np.mean(masked_errors)
    unmasked_mean = np.mean(unmasked_errors)
    masked_std = np.std(masked_errors)
    unmasked_std = np.std(unmasked_errors)
    
    ax2.axhline(masked_mean, color='#FF6F00', linestyle='--', linewidth=2.5, alpha=0.7,
                label=f'Masked Mean: {masked_mean:.4f}')
    ax2.fill_between([steps[0], steps[-1]], 
                     masked_mean - masked_std, masked_mean + masked_std,
                     alpha=0.15, color='orange', label='±1 std')
    
    ax2.axhline(unmasked_mean, color='#2E7D32', linestyle='--', linewidth=2.5, alpha=0.7,
                label=f'Unmasked Mean: {unmasked_mean:.4f}')
    ax2.fill_between([steps[0], steps[-1]], 
                     unmasked_mean - unmasked_std, unmasked_mean + unmasked_std,
                     alpha=0.15, color='green')
    
    ax2.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=13, fontweight='bold')
    ax2.set_title('Reconstruction Error: Masked vs Unmasked', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
    ax2.tick_params(axis='both', labelsize=11)
    
    # Add improvement ratio (larger, clearer)
    if unmasked_errors[0] > 0:
        improvement = (unmasked_errors[0] - unmasked_errors[-1]) / unmasked_errors[0] * 100
    else:
        improvement = 0
    if masked_errors[0] > 0:
        masked_improvement = (masked_errors[0] - masked_errors[-1]) / masked_errors[0] * 100
    else:
        masked_improvement = 0
    
    improvement_text = f'Improvement:\nUnmasked: {improvement:+.1f}%\nMasked: {masked_improvement:+.1f}%'
    ax2.text(0.98, 0.02, improvement_text, transform=ax2.transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85, 
                      edgecolor='black', linewidth=1.5),
             fontsize=11, family='monospace', fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def plot_pose_reconstruction_comparison(masked, recon, target, obs_mask, edges=None, 
                                       keypoint_names=None, title="Pose Reconstruction", max_samples=4):
    """
    Simplified, clearer pose visualization focused on reconstruction quality.
    Shows before/after comparison in 2D projections only (easier to interpret).
    
    Args:
        masked: (B, 3, 13) masked input poses
        recon: (B, 3, 13) reconstructed poses
        target: (B, 3, 13) target poses
        obs_mask: (B, 13) observation mask
        edges: skeleton edges
        keypoint_names: keypoint names
        title: plot title
        max_samples: maximum number of samples to show
    """
    B = min(masked.shape[0], max_samples)
    
    # Convert to numpy
    masked_np = masked.cpu().numpy() if hasattr(masked, 'cpu') else masked
    recon_np = recon.cpu().numpy() if hasattr(recon, 'cpu') else recon
    target_np = target.cpu().numpy() if hasattr(target, 'cpu') else target
    mask_np = obs_mask.cpu().numpy() if hasattr(obs_mask, 'cpu') else obs_mask
    
    # Use 2D projections (easier to see)
    projections = [('XY (Top View)', [0, 1]), ('XZ (Side View)', [0, 2]), ('YZ (Front View)', [1, 2])]
    
    # Larger figure for better visibility
    fig, axes = plt.subplots(B, 3, figsize=(18, 6*B), dpi=100)
    if B == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(B):
        m = masked_np[i]
        r = recon_np[i]
        t = target_np[i]
        mask = mask_np[i].astype(bool)
        
        for j, (proj_name, (ax1, ax2)) in enumerate(projections):
            ax = axes[i, j]
            
            # Plot target (ground truth) - gray, largest
            ax.scatter(t[ax1], t[ax2], s=150, marker='o', c='gray', 
                      edgecolors='black', linewidths=2, label='Target', alpha=0.7, zorder=3)
            
            # Plot masked input - orange X for masked, blue circle for observed
            masked_coords = m[ax1][~mask]
            masked_coords2 = m[ax2][~mask]
            observed_coords = m[ax1][mask]
            observed_coords2 = m[ax2][mask]
            
            if len(masked_coords) > 0:
                ax.scatter(masked_coords, masked_coords2, s=200, marker='x', 
                          c='orange', linewidths=3, label='Masked Input', zorder=4)
            if len(observed_coords) > 0:
                ax.scatter(observed_coords, observed_coords2, s=100, marker='o',
                          c='blue', edgecolors='darkblue', linewidths=1.5, 
                          label='Observed Input', alpha=0.6, zorder=2)
            
            # Plot reconstruction - green stars
            ax.scatter(r[ax1], r[ax2], s=120, marker='*', c='green', 
                      edgecolors='darkgreen', linewidths=1.5, label='Reconstruction', zorder=5)
            
            # Draw error vectors for masked points only
            for k in np.where(~mask)[0]:
                ax.plot([r[ax1, k], t[ax1, k]], [r[ax2, k], t[ax2, k]], 
                       linestyle='--', linewidth=2, alpha=0.5, color='red', zorder=1)
            
            # Skeleton connections on target
            if edges is not None:
                for (a, b) in edges:
                    ax.plot([t[ax1, a], t[ax1, b]], [t[ax2, a], t[ax2, b]], 
                           linewidth=1.5, alpha=0.3, color='gray', linestyle=':', zorder=0)
            
            # Labels and title
            ax.set_title(f'Sample {i+1}: {proj_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel(f'{["X", "X", "Y"][j]} (mm)', fontsize=10)
            ax.set_ylabel(f'{["Y", "Z", "Z"][j]} (mm)', fontsize=10)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            
            # Legend only on first plot
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    
    # Add overall statistics
    errors = np.linalg.norm(recon_np[:B] - target_np[:B], axis=1)  # (B, 13)
    overall_rmse = np.sqrt(np.mean((recon_np[:B] - target_np[:B])**2))
    
    stats_text = f'Average RMSE: {overall_rmse:.3f}\nPer-sample RMSE: {np.mean(errors, axis=1)}'
    fig.text(0.5, 0.02, f'Reconstruction Quality: {overall_rmse:.3f} mm RMSE', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    return fig

