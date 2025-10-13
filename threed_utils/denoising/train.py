from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)

    args = parser.parse_args()
    data_path: Path = args.data_path

    poses = np.load(data_path)
    train_data = KeypointDataset(poses)
    masked, target = train_data[0]
    print(f"Masked shape: {masked.shape}, Target shape: {target.shape}")
