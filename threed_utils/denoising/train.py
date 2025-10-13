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


def mask_keypoints(pose, k_min=1, k_max=None, fill=0.0, rng=None):
    """
    pose: (space=3, keypoints=13, individuals=1)
    Masks a random number of keypoints along the keypoint axis.
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

    pose[:, idx, :] = fill if keypoint_axis == 1 else pose[:, :, idx]
    return pose, idx


class KeypointDataset(Dataset):
    def __init__(self, poses, k_min=1, k_max=None, fill=0.0):
        """
        poses: array of shape (frames, space=3, keypoints=13, individuals=1)
        """
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
            verbose=True,
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

        self.writer.close()


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
    full_ds = KeypointDataset(poses, k_min=args.k_min, k_max=args.k_max, fill=args.fill)
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
