import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from threed_utils.denoising.temporal_DAE import TemporalDAE, build_mask, flatten_pose, loss_masked
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
    K = pose.shape[1]  # keypoints axis = 1 in (3,13,1)
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


class TemporalKeypointDataset(Dataset):
    """
    Dataset that returns temporal windows of poses for the TemporalDAE.
    For each sample at index t, returns frames [t-w, ..., t, ..., t+w]
    """
    def __init__(self, poses, window_size=2, k_min=1, k_max=None, fill=0.0):
        """
        poses: array of shape (frames, space=3, keypoints=13, individuals=1)
        window_size: w in [t-w, t+w]
        """
        self.poses = np.asarray(poses).astype(np.float32)
        self.window_size = window_size
        self.k_min, self.k_max, self.fill = k_min, k_max, fill
        
        # Valid indices are those with full temporal context
        self.n_frames = self.poses.shape[0]
        self.valid_indices = list(range(window_size, self.n_frames - window_size))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get center frame index
        center_idx = self.valid_indices[idx]
        
        # Extract temporal window
        start_idx = center_idx - self.window_size
        end_idx = center_idx + self.window_size + 1  # +1 because python slicing
        
        window_poses = self.poses[start_idx:end_idx]  # (2w+1, 3, 13, 1)
        
        # Mask only the CENTER frame
        target_center = window_poses[self.window_size].copy()  # (3, 13, 1)
        masked_center, _ = mask_keypoints(target_center, self.k_min, self.k_max, self.fill)
        
        # Replace center with masked version
        window_poses_masked = window_poses.copy()
        window_poses_masked[self.window_size] = masked_center
        
        # Flatten each frame in the window
        # window_poses_masked: (2w+1, 3, 13, 1) -> (2w+1, 39)
        window_flat = window_poses_masked.reshape(window_poses_masked.shape[0], -1)
        target_flat = target_center.reshape(-1)  # (39,)
        
        return torch.from_numpy(window_flat), torch.from_numpy(target_flat)


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
    window_size: int = 2
    temporal_model: str = "transformer"  # 'transformer', 'lstm', or 'conv1d'
    device: str = "cuda" if torch.cuda.is_available() else "mps"


class TemporalAutoEncoderTrainer:
    def __init__(self, args: TrainerArgs):
        self.args = args
        self.device = args.device
        self.model = TemporalDAE(
            d=39, 
            h=args.latent_dim, 
            window_size=args.window_size,
            temporal_model=args.temporal_model
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
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
        self.vis_x = xb[:4]  # show up to 4 windowed poses
        self.vis_y = yb[:4]
        
        # Define keypoint names and skeleton edges
        self.keypoint_names = [
            'nose', 'ear_lf', 'forepaw_lf', 'hindpaw_lf', 'tailbase',
            'hindpaw_rt', 'forepaw_rt', 'ear_rt', 'belly_rostral', 
            'belly_caudal', 'back_caudal', 'back_mid', 'back_rostral'
        ]
        
        self.edges = [
            (0, 1), (0, 7), (1, 7), (1, 12), (7, 12), (12, 11),
            (11, 10), (10, 4), (1, 8), (7, 6), (8, 2), (8, 6),
            (2, 9), (9, 3), (9, 6), (9, 4), (5, 9)
        ]
        
        # Track keypoint errors over time for progression plots
        self.keypoint_error_history = [] 


    def training_step(self, x_window: Tensor, y: Tensor) -> float:
        """
        x_window: (B, 2w+1, 39) - temporal window of flattened poses
        y: (B, 39) - target center frame
        """
        x_window, y = x_window.to(self.device), y.to(self.device)
        
        # Build mask for each frame in the window
        B, n_frames, d = x_window.shape
        m_window = torch.zeros_like(x_window)
        for i in range(n_frames):
            # Reshape to (B, 3, 13, 1) for build_mask
            x_frame = x_window[:, i, :].view(B, 3, 13, 1)
            m_frame = build_mask(x_frame, fill=self.args.fill)  # (B, 39)
            m_window[:, i, :] = m_frame
        
        self.optimizer.zero_grad(set_to_none=True)
        pred, _ = self.model(x_window, m_window)
        
        # Loss only on center frame
        center_m = m_window[:, self.args.window_size, :]  # (B, 39)
        loss = self.criterion(pred, y, center_m)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def test_step(self, x_window: Tensor, y: Tensor) -> float:
        x_window, y = x_window.to(self.device), y.to(self.device)
        
        # Build mask for each frame in the window
        B, n_frames, d = x_window.shape
        m_window = torch.zeros_like(x_window)
        for i in range(n_frames):
            x_frame = x_window[:, i, :].view(B, 3, 13, 1)
            m_frame = build_mask(x_frame, fill=self.args.fill)
            m_window[:, i, :] = m_frame
        
        with torch.no_grad():
            pred, _ = self.model(x_window, m_window)
            center_m = m_window[:, self.args.window_size, :]
            loss = self.criterion(pred, y, center_m)
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

            print(f"Epoch {epoch + 1}/{self.args.epochs} | Train: {avg_train:.4f} | Test: {avg_test:.4f} | LR: {lr:.2e}")
            self.writer.flush()
            if (epoch + 1) % 10 == 0:
                self.log_visuals(epoch+1)

        self.writer.close()
        
    @torch.no_grad()
    def log_visuals(self, step: int):
        """Visualize reconstruction for center frame"""
        x_window = self.vis_x.to(self.device)  # (B, 2w+1, 39)
        y = self.vis_y.to(self.device)  # (B, 39)
        
        # Build masks
        B, n_frames, d = x_window.shape
        m_window = torch.zeros_like(x_window)
        for i in range(n_frames):
            x_frame = x_window[:, i, :].view(B, 3, 13, 1)
            m_frame = build_mask(x_frame, fill=self.args.fill)
            m_window[:, i, :] = m_frame
        
        pred_f, _ = self.model(x_window, m_window)
        
        # Extract center frame for visualization
        center_idx = self.args.window_size
        center_x = x_window[:, center_idx, :].view(B, 3, 13)  # (B, 3, 13)
        center_m = m_window[:, center_idx, :].view(B, 3, 13)[:, 0, :]  # (B, 13)
        
        target = y.view(B, 3, 13)  # (B, 3, 13)
        recon = pred_f.view(B, 3, 13)  # (B, 3, 13)
        
        obs_mask = center_m  # (B, 13)
        
        # Calculate per-keypoint errors
        err = torch.sqrt(((recon - target) ** 2).mean(dim=0))  # (3, 13)
        keypoint_errors = torch.norm(err, dim=0).cpu().numpy()  # (13,)
        self.keypoint_error_history.append(keypoint_errors)
        
        # Import plotting functions from train.py
        from threed_utils.denoising.train import (
            _plot_pose3d_grid, _plot_pose2d_projections,
            _plot_keypoint_error_bar, _plot_keypoint_error_progression
        )
        
        # 3D grid
        fig3d = _plot_pose3d_grid(center_x, recon, target, obs_mask, edges=self.edges,
                                 keypoint_names=self.keypoint_names, title="Temporal Pose3D")
        self.writer.add_figure("Viz/Pose3D", fig3d, global_step=step)
        plt.close(fig3d)
        
        # 2D projections
        fig2d = _plot_pose2d_projections(center_x, recon, target, obs_mask, edges=self.edges,
                                        keypoint_names=self.keypoint_names, title="Temporal Pose2D")
        self.writer.add_figure("Viz/Pose2D", fig2d, global_step=step)
        plt.close(fig2d)
        
        # Per-keypoint RMSE
        figbar = _plot_keypoint_error_bar(err.cpu().numpy().T, keypoint_names=self.keypoint_names)
        self.writer.add_figure("Viz/KeypointError", figbar, global_step=step)
        plt.close(figbar)
        
        # Error progression
        if len(self.keypoint_error_history) > 1:
            figprog = _plot_keypoint_error_progression(self.keypoint_error_history,
                                                     keypoint_names=self.keypoint_names)
            self.writer.add_figure("Viz/KeypointErrorProgression", figprog, global_step=step)
            plt.close(figprog)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=Path("./logs_temporal"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--fill", type=float, default=0.0)
    parser.add_argument("--window_size", type=int, default=2,
                       help="Temporal window size (w). Uses [t-w, t+w] frames")
    parser.add_argument("--temporal_model", type=str, default="transformer",
                       choices=["transformer", "lstm", "conv1d"],
                       help="Type of temporal model to use")
    args = parser.parse_args()

    poses = np.load(args.data_path).astype(np.float32)  # (N,3,13,1)

    print(f"Dataset: {poses.shape[0]} frames total")
    print(f"Window size: {args.window_size} (using {2*args.window_size+1} frames per sample)")
    print(f"Temporal model: {args.temporal_model}")
    
    full_ds = TemporalKeypointDataset(
        poses, window_size=args.window_size,
        k_min=args.k_min, k_max=args.k_max, fill=args.fill
    )
    
    print(f"Valid samples (with full temporal context): {len(full_ds)}")
    
    # train/val split (90/10) based on actual dataset length
    N = len(full_ds)
    n_train = int(0.9 * N)
    n_val = N - n_train
    
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
        window_size=args.window_size,
        temporal_model=args.temporal_model,
        device="cuda" if torch.cuda.is_available() else "mps",
    )

    trainer = TemporalAutoEncoderTrainer(trainer_args)
    trainer.train()

