from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class KeypointDataset(Dataset):
    def __init__(self, poses):
        self.poses = poses

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        pose = self.poses[idx]

        return pose


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)

    args = parser.parse_args()
    data_path: Path = args.data_path

    poses = np.load(data_path)
    train_data = KeypointDataset(poses)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # test for loader:
    for batch in tqdm(train_loader):
        print(batch.shape)
