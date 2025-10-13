import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm


def score_poses(poses: xr.Dataset, t: float = 0.4, cutoff: int = 5):
    """Function to score frames based on number of keypoints >= t"""

    binmask = (poses["confidence"] >= t).astype("int8")
    # score
    score = binmask.sum(dim="keypoints").rename("score")

    if poses.dims.get("individuals", 1) == 1:
        score_time = score.squeeze("individuals", drop=True)
    else:
        score_time = score
    return score_time


def compute_stats_score(
    score_time: xr.Dataset, poses: xr.Dataset, quantile_list: List = [0.75, 0.9, 0.95]
):
    """Computes and returns mean, std, quantiles of scores per file"""
    vals = score_time.values.ravel()
    counts = np.bincount(vals, minlength=poses.dims["keypoints"])
    print({i: int(counts[i]) for i in range(len(counts)) if counts[i]})

    print("mean:", float(vals.mean()), "std:", float(vals.std()))
    print("quantiles:", np.quantile(vals, [0, 0.25, 0.5, 0.75, 0.9, 0.95]).tolist())
    return vals.mean(), vals.std(), np.quantile(vals, quantile_list).tolist()


def load_poses(file_path: Path) -> torch.Tensor:
    """Function to load the poses into a tensor"""

    try:
        poses = xr.load_dataset(file_path)
    except FileNotFoundError:
        print(f"{file_path} not found")

    # we only take the coordinates:
    positions = poses.position.values
    pass


if __name__ == "__main__":
    print("hello")
