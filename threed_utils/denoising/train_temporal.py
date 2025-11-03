import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from threed_utils.denoising.temporal_DAE import TemporalDAE, build_mask, loss_masked
from threed_utils.denoising.visualization import (
    plot_pose_reconstruction_comparison,
    plot_keypoint_error_bar, plot_keypoint_error_progression
)
from threed_utils.denoising.metrics import compute_all_metrics
from threed_utils.denoising.constants import KEYPOINT_NAMES, SKELETON_EDGES
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


def mask_keypoints(pose, k_min=1, k_max=None, fill=0.0, rng=None, mask_strategy='zero', 
                   noise_level=0.2):
    """
    pose: (space=3, keypoints=13, individuals=1)
    Mask k keypoints - always zeros them out, optionally adds aggressive noise to context.
    
    Args:
        mask_strategy: 'zero' (clean masking) or 'gaussian' (mask + add aggressive noise)
        noise_level: std of gaussian noise relative to pose scale (default 0.2 = 20%)
    
    Returns:
        masked_pose, masked_indices
    """
    pose = np.asarray(pose).copy()
    
    # Determine keypoint axis
    if pose.shape[1] == 13:
        keypoint_axis = 1
    else:
        keypoint_axis = 2

    if k_max is None:
        k_max = pose.shape[keypoint_axis]

    if rng is None:
        rng = np.random

    # Select keypoints to mask
    m = rng.randint(k_min, k_max + 1)
    idx = rng.choice(pose.shape[keypoint_axis], size=m, replace=False)
    
    # ALWAYS zero out selected keypoints (this is what we need to reconstruct)
    if keypoint_axis == 1:
        pose[:, idx, :] = fill
    else:
        pose[:, :, idx] = fill
    
    # For gaussian strategy: add aggressive noise to UNMASKED keypoints
    # This forces the model to learn robust representations
    if mask_strategy == 'gaussian':
        pose_scale = np.std(pose[pose != 0]) if np.any(pose != 0) else 1.0
        noise_std = noise_level * pose_scale
        
        # Create mask of unmasked keypoints
        all_idx = np.arange(pose.shape[keypoint_axis])
        unmasked_idx = np.setdiff1d(all_idx, idx)
        
        if len(unmasked_idx) > 0:
            if keypoint_axis == 1:
                noise = rng.normal(0, noise_std, (3, len(unmasked_idx), 1))
                pose[:, unmasked_idx, :] += noise
            else:
                noise = rng.normal(0, noise_std, (3, 1, len(unmasked_idx)))
                pose[:, :, unmasked_idx] += noise

    return pose, idx


class TemporalKeypointDataset(Dataset):
    """
    Dataset that returns temporal windows for TemporalDAE.
    Center frame has keypoints masked (zeroed). Gaussian strategy adds noise to context.
    """
    def __init__(self, poses, window_size=2, k_min=1, k_max=None, fill=0.0, 
                 mask_strategy='zero', noise_level=0.2):
        """
        poses: (frames, 3, 13, 1)
        mask_strategy: 'zero' (clean) or 'gaussian' (+ aggressive noise on context frames)
        noise_level: 0.2 = 20% of pose std
        """
        self.poses = np.asarray(poses).astype(np.float32)
        self.window_size = window_size
        self.k_min, self.k_max, self.fill = k_min, k_max, fill
        self.mask_strategy = mask_strategy
        self.noise_level = noise_level
        self.n_frames = self.poses.shape[0]
        self.valid_indices = list(range(window_size, self.n_frames - window_size))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        start_idx = center_idx - self.window_size
        end_idx = center_idx + self.window_size + 1
        
        window_poses = self.poses[start_idx:end_idx].copy()  # (2w+1, 3, 13, 1)
        target_center = window_poses[self.window_size].copy()  # (3, 13, 1) - clean
        
        # Mask center frame (ALWAYS zeros out selected keypoints)
        masked_center, masked_idx = mask_keypoints(
            window_poses[self.window_size], self.k_min, self.k_max, self.fill,
            mask_strategy='zero', noise_level=self.noise_level  # center always clean
        )
        window_poses[self.window_size] = masked_center
        
        # For gaussian: add aggressive noise to CONTEXT frames (not center)
        if self.mask_strategy == 'gaussian':
            pose_scale = np.std(self.poses[self.poses != 0])
            noise_std = self.noise_level * pose_scale
            
            for i in range(len(window_poses)):
                if i != self.window_size:  # Skip center
                    noise = np.random.normal(0, noise_std, window_poses[i].shape)
                    window_poses[i] += noise
        
        window_flat = window_poses.reshape(len(window_poses), -1)  # (2w+1, 39)
        target_flat = target_center.reshape(-1)  # (39,)
        
        return torch.from_numpy(window_flat), torch.from_numpy(target_flat)


@dataclass
class TrainerArgs:
    dataset: Dataset
    testset: Dataset
    log_dir: Path = Path("./logs")
    latent_dim: int = 128
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
        
        self.keypoint_names = KEYPOINT_NAMES
        self.edges = SKELETON_EDGES
        
        # Track keypoint errors over time for progression plots
        self.keypoint_error_history = [] 


    def training_step(self, x_window, y) -> float:
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

    def test_step(self, x_window, y) -> float:
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

            # Log detailed metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.log_visuals(epoch+1)
                
                # Compute and log comprehensive metrics
                with torch.no_grad():
                    x_test, y_test = next(iter(self.test_loader))
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                    B, n_frames, d = x_test.shape
                    
                    m_window = torch.zeros_like(x_test)
                    for i in range(n_frames):
                        x_frame = x_test[:, i, :].view(B, 3, 13, 1)
                        m_frame = build_mask(x_frame, fill=self.args.fill)
                        m_window[:, i, :] = m_frame
                    
                    pred, _ = self.model(x_test, m_window)
                    center_m = m_window[:, self.args.window_size, :]
                    pred_3d = pred.view(B, 3, 13)
                    target_3d = y_test.view(B, 3, 13)
                    mask_2d = center_m.view(B, 3, 13)[:, 0, :]
                    
                    metrics = compute_all_metrics(pred_3d, target_3d, mask_2d)
                    
                    # Log key metrics to TensorBoard
                    self.writer.add_scalar("Metrics/RMSE", metrics['rmse'], epoch)
                    self.writer.add_scalar("Metrics/MAE", metrics['mae'], epoch)
                    if 'rmse_masked' in metrics:
                        self.writer.add_scalar("Metrics/RMSE_Masked", metrics['rmse_masked'], epoch)
                    if 'rmse_unmasked' in metrics:
                        self.writer.add_scalar("Metrics/RMSE_Unmasked", metrics['rmse_unmasked'], epoch)
            
            print(f"Epoch {epoch + 1}/{self.args.epochs} | Train: {avg_train:.4f} | Test: {avg_test:.4f} | LR: {lr:.2e}")
            self.writer.flush()

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
        
        # Reconstruction visualization
        fig_2d = plot_pose_reconstruction_comparison(
            center_x, recon, target, obs_mask, edges=self.edges,
            keypoint_names=self.keypoint_names, title="Pose Reconstruction - 2D Views", max_samples=4
        )
        self.writer.add_figure("Viz/PoseReconstruction2D", fig_2d, global_step=step)
        plt.close(fig_2d)
        
        # Per-keypoint RMSE
        figbar = plot_keypoint_error_bar(err.cpu().numpy().T, keypoint_names=self.keypoint_names)
        self.writer.add_figure("Viz/KeypointError", figbar, global_step=step)
        plt.close(figbar)
        
        # Error progression
        if len(self.keypoint_error_history) > 1:
            figprog = plot_keypoint_error_progression(self.keypoint_error_history,
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
    parser.add_argument("--mask_strategy", type=str, default="zero",
                       choices=["zero", "gaussian", "dropout", "swap", "mixed"],
                       help="Masking strategy for data augmentation")
    parser.add_argument("--noise_level", type=float, default=0.2,
                       help="Noise level for gaussian masking (std as fraction of pose scale)")
    args = parser.parse_args()

    poses = np.load(args.data_path).astype(np.float32)  # (N,3,13,1)

    print(f"Dataset: {poses.shape[0]} frames total")
    print(f"Window size: {args.window_size} (using {2*args.window_size+1} frames per sample)")
    print(f"Temporal model: {args.temporal_model}")
    
    full_ds = TemporalKeypointDataset(
        poses, window_size=args.window_size,
        k_min=args.k_min, k_max=args.k_max, fill=args.fill,
        mask_strategy=args.mask_strategy, noise_level=args.noise_level
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

