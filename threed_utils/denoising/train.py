import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from threed_utils.denoising.DAE import DAE, build_mask, flatten_pose, loss_masked
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# plotting 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mask_keypoints(pose, k_min=1, k_max=None, fill=0.0, rng=None):
    """
    pose: (space=3, keypoints=13, individuals=1)
    Randomly mask k keypoints by setting them to fill value.
    """
    pose = np.asarray(pose).copy()
    K = pose.shape[1]  # keypoints axis = 1 in (3,13,1)? -> careful: shape is (3,13,1)
    K = pose.shape[1] if pose.shape[1] == 13 else pose.shape[2]

    if pose.shape[1] == 13:  # (frames, 3, 13, 1)
        keypoint_axis = 1
    else:  # (frames, 3, 13, 1) squeezed wrongly
        keypoint_axis = 2

    if k_max is None:
        k_max = pose.shape[keypoint_axis]

    if rng is None:
        rng = np.random

    m = rng.randint(k_min, k_max + 1)
    idx = rng.choice(pose.shape[keypoint_axis], size=m, replace=False)
    
    if keypoint_axis == 1:
        pose[:, idx, :] = fill
    else:
        pose[:, :, idx] = fill

    return pose, idx


class KeypointDataset(Dataset):
    def __init__(self, poses, k_min=1, k_max=None, fill=0.0):
        self.poses = np.asarray(poses).astype(np.float32)
        self.k_min, self.k_max, self.fill = k_min, k_max, fill

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        target = self.poses[idx]  # (3, 13, 1)
        masked, _ = mask_keypoints(target, self.k_min, self.k_max, self.fill)
        return torch.from_numpy(masked), torch.from_numpy(target)


@dataclass
class TrainerArgs:
    dataset: Dataset
    testset: Dataset
    log_dir: Path = Path("./logs")
    latent_dim: int = 128
    hidden_dim_size: int = 256
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    k_min: int = 1
    k_max: int = 5
    fill: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "mps"


class AutoEncoderTrainer:
    def __init__(self, args: TrainerArgs):
        self.args = args
        self.device = args.device
        self.model = DAE(d=39, h=args.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-3,
            cooldown=1,
            min_lr=1e-6,
        )
        self.criterion = loss_masked
        self.train_loader = DataLoader(
            args.dataset, batch_size=args.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            args.testset, batch_size=args.batch_size, shuffle=False
        )

        run_dir = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(run_dir), flush_secs=1)
        self.global_step = 0
        xb, yb = next(iter(self.test_loader))
        self.vis_x = xb[:4]  # show up to 4 poses
        self.vis_y = yb[:4]
        
        # Define keypoint names and skeleton edges
        self.keypoint_names = [
            'nose', 'ear_lf', 'forepaw_lf', 'hindpaw_lf', 'tailbase',
            'hindpaw_rt', 'forepaw_rt', 'ear_rt', 'belly_rostral', 
            'belly_caudal', 'back_caudal', 'back_mid', 'back_rostral'
        ]
        
        # Define skeleton edges (connections between keypoints)
        self.edges = [
            (0, 1), (0, 7), (1, 7), (1, 12), (7, 12), (12, 11), 
            (11, 10), (10, 4), (1, 8), (7, 6), (8, 2), (8, 6), 
            (2, 9), (9, 3), (9, 6), (9, 4), (5, 9)
        ]
        
        # Track keypoint errors over time for progression plots
        self.keypoint_error_history = [] 


    def training_step(self, x: Tensor, y: Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        x_flat, y_flat = flatten_pose(x), flatten_pose(y)
        m_flat = build_mask(x, fill=self.args.fill).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        pred, _ = self.model(x_flat, m_flat)
        loss = self.criterion(pred, y_flat, m_flat)  # normalized masked loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def test_step(self, x: Tensor, y: Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        x_flat, y_flat = flatten_pose(x), flatten_pose(y)
        m_flat = build_mask(x, fill=self.args.fill).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(x_flat, m_flat)
            loss = self.criterion(pred, y_flat, m_flat)
        return float(loss.item())

    def train(self):
        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []
            for x, y in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.args.epochs} - Training",
            ):
                loss = self.training_step(x, y)
                train_losses.append(loss)
                self.writer.add_scalar("Loss/TrainStep", loss, self.global_step)
                self.global_step += 1

            avg_train = float(np.mean(train_losses))
            self.writer.add_scalar("Loss/Train", avg_train, epoch)

            self.model.eval()
            test_losses = []
            for x, y in tqdm(
                self.test_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs} - Testing"
            ):
                loss = self.test_step(x, y)
                test_losses.append(loss)
            avg_test = float(np.mean(test_losses))
            self.writer.add_scalar("Loss/Test", avg_test, epoch)

            # LR schedule on plateau + log LR
            self.scheduler.step(avg_test)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", lr, epoch)

            print(
                f"Epoch {epoch + 1}/{self.args.epochs} | Train: {avg_train:.4f} | Test: {avg_test:.4f} | LR: {lr:.2e}"
            )
            self.writer.flush()
            if (epoch + 1) % 10 == 0:
                self.log_visuals(epoch+1)

        self.writer.close()
    @torch.no_grad()
    def log_visuals(self, step: int):
        x = self.vis_x.to(self.device)             # (B,3,13,1)
        y = self.vis_y.to(self.device)
        x_flat = flatten_pose(x)
        m_flat  = build_mask(x, fill=self.args.fill).to(self.device)  # (B,39) 1=observed
        pred_f, _ = self.model(x_flat, m_flat)

        masked = x.squeeze(-1)                     # (B,3,13)
        target = y.squeeze(-1)                     # (B,3,13)
        recon  = pred_f.view(-1, 3, 13)            # (B,3,13)
        obs_mask = m_flat.view(-1, 3, 13)[:,0,:]   # (B,13) take one axis (all same)

        # Calculate per-keypoint errors and store for progression tracking
        err = torch.sqrt(((recon - target) ** 2).mean(dim=0))  # (3,13)
        keypoint_errors = torch.norm(err, dim=0).cpu().numpy()  # (13,) - overall RMSE per keypoint
        self.keypoint_error_history.append(keypoint_errors)

        # 3D grid with observed/masked markers + recon→target arrows
        fig3d = _plot_pose3d_grid(masked, recon, target, obs_mask, edges=self.edges, 
                                 keypoint_names=self.keypoint_names, title="Pose3D (obs vs masked)")
        self.writer.add_figure("Viz/Pose3D", fig3d, global_step=step)
        plt.close(fig3d)

        # 2D projections for better visualization
        fig2d = _plot_pose2d_projections(masked, recon, target, obs_mask, edges=self.edges,
                                        keypoint_names=self.keypoint_names, title="Pose2D Projections")
        self.writer.add_figure("Viz/Pose2D", fig2d, global_step=step)
        plt.close(fig2d)

        # per-keypoint RMSE with names (averaged over batch)
        figbar = _plot_keypoint_error_bar(err.cpu().numpy().T, keypoint_names=self.keypoint_names)
        self.writer.add_figure("Viz/KeypointError", figbar, global_step=step)
        plt.close(figbar)

        # Keypoint error progression over time
        if len(self.keypoint_error_history) > 1:
            figprog = _plot_keypoint_error_progression(self.keypoint_error_history, 
                                                     keypoint_names=self.keypoint_names)
            self.writer.add_figure("Viz/KeypointErrorProgression", figprog, global_step=step)
            plt.close(figprog)

def _set_equal_3d(ax):
    xs, ys, zs = [getattr(ax, f"get_{a}lim")() for a in "xyz"]
    ranges = np.array([xs, ys, zs])
    minv, maxv = ranges[:,0].min(), ranges[:,1].max()
    center = (minv + maxv) / 2.0
    span = (maxv - minv)
    half = span / 2.0
    ax.set_xlim(center-half, center+half)
    ax.set_ylim(center-half, center+half)
    ax.set_zlim(center-half, center+half)

def _plot_pose3d_grid(masked, recon, target, obs_mask, edges=None, keypoint_names=None, title="Pose3D"):
    """
    masked/recon/target: (B,3,13) ; obs_mask: (B,13) with 1=observed, 0=masked
    edges: list of (i,j) pairs to draw skeleton (optional)
    keypoint_names: list of keypoint names for labeling
    """
    B = masked.shape[0]
    cols, rows = 2, int(np.ceil(B/2))
    fig = plt.figure(figsize=(cols*6, rows*6))
    for i in range(B):
        ax = fig.add_subplot(rows, cols, i+1, projection="3d")
        m, r, t = masked[i].cpu().numpy(), recon[i].cpu().numpy(), target[i].cpu().numpy()
        mask = obs_mask[i].cpu().numpy().astype(bool)

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
            ax.plot([r[0,j], t[0,j]], [r[1,j], t[1,j]], [r[2,j], t[2,j]], 
                   linestyle="--", linewidth=2, alpha=0.7, color="purple")

        # skeleton connections
        if edges is not None:
            for (a,b) in edges:
                ax.plot(t[0,[a,b]], t[1,[a,b]], t[2,[a,b]], 
                       linewidth=2, alpha=0.4, color="gray")

        ax.set_title(f"Sample {i}", fontsize=12)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        _set_equal_3d(ax)
        if i == 0: 
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

def _plot_keypoint_error_bar(err_xyz, keypoint_names=None, title="Keypoint RMSE (sorted)"):
    """
    err_xyz: (13,3) per-axis RMSE; will show overall (L2 over xyz) sorted desc
    keypoint_names: list of keypoint names for labeling
    """
    rmse = np.linalg.norm(err_xyz, axis=1)  # (13,)
    order = np.argsort(-rmse)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars with colors based on error magnitude
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, 13))
    bars = ax.bar(np.arange(13), rmse[order], color=colors)
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, rmse[order])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
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

def _plot_pose2d_projections(masked, recon, target, obs_mask, edges=None, keypoint_names=None, title="Pose2D Projections"):
    """
    Create 2D projections (XY, XZ, YZ) for better visualization
    masked/recon/target: (B,3,13) ; obs_mask: (B,13) with 1=observed, 0=masked
    """
    B = masked.shape[0]
    projections = [('XY', [0, 1]), ('XZ', [0, 2]), ('YZ', [1, 2])]
    
    fig, axes = plt.subplots(B, 3, figsize=(15, 5*B))
    if B == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(B):
        m, r, t = masked[i].cpu().numpy(), recon[i].cpu().numpy(), target[i].cpu().numpy()
        mask = obs_mask[i].cpu().numpy().astype(bool)
        
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
                ax.plot([r[ax1,k], t[ax1,k]], [r[ax2,k], t[ax2,k]], 
                       linestyle="--", linewidth=2, alpha=0.7, color="purple")
            
            # skeleton connections
            if edges is not None:
                for (a,b) in edges:
                    ax.plot(t[ax1,[a,b]], t[ax2,[a,b]], 
                           linewidth=2, alpha=0.4, color="gray")
            
            ax.set_title(f"Sample {i} - {proj_name}")
            ax.set_xlabel(f"{proj_name[0]}"); ax.set_ylabel(f"{proj_name[1]}")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0: 
                ax.legend(fontsize=8, loc="upper left")
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

def _plot_keypoint_error_progression(error_history, keypoint_names=None, title="Keypoint Error Progression"):
    """
    Plot the progression of keypoint errors over training steps
    error_history: list of arrays, each of shape (13,) - RMSE per keypoint
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=Path("./logs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--fill", type=float, default=0.0)
    args = parser.parse_args()

    poses = np.load(args.data_path).astype(np.float32)  # (N,3,13,1)

    # train/val split (90/10)
    N = poses.shape[0]
    n_train = int(0.9 * N)
    n_val = N - n_train
    full_ds = KeypointDataset(
        poses, k_min=args.k_min, k_max=args.k_max, fill=args.fill
    )
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    trainer_args = TrainerArgs(
        dataset=train_ds,
        testset=val_ds,
        log_dir=args.log_dir,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_min=args.k_min,
        k_max=args.k_max,
        fill=args.fill,
        device="cuda" if torch.cuda.is_available() else "mps",
    )

    trainer = AutoEncoderTrainer(trainer_args)
    trainer.train()
