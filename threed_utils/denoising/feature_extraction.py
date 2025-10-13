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


def select_top_frames(
    poses: xr.Dataset,
    threshold: float,
    quantile: float,
    max_frames: int,
    individual: str | None = None,
):
    """
    Returns (selected_positions, selected_times, cutoff, indices)
    - selected_positions: (time, space, keypoints[, individuals])
    - selected_times:     time coords of selected frames
    - cutoff:             int score cutoff used
    - indices:            np.ndarray of selected time indices (in original timeline)
    """
    # 1) score per frame (uses your helper)
    score = score_poses(poses, t=threshold)  # (time) or (time, individuals)

    # 2) handle individuals briefly
    if "individuals" in score.dims:
        if individual is None:
            if poses.sizes["individuals"] != 1:
                raise ValueError(
                    "Specify `individual` when multiple individuals exist."
                )
            individual = poses.individuals.item()
        score_i = score.sel(individuals=individual)
        pos = poses["position"].sel(individuals=individual)
    else:
        score_i = score
        pos = poses["position"]

    # 3) cutoff from quantile
    cutoff = int(np.ceil(np.quantile(score_i.values, quantile)))

    # 4) keep frames ≥ cutoff, sort by score desc then time asc, take top n
    vals = score_i.values
    idx = np.flatnonzero(vals >= cutoff)
    if idx.size:
        order = np.lexsort(
            (-vals[idx], idx)
        )  # primary: score desc, secondary: time asc
        top = idx[order][:max_frames]
    else:
        top = idx  # empty

    # 5) select
    top_da = xr.DataArray(top, dims="time")
    selected_positions = pos.isel(time=top_da)
    selected_times = poses["time"].isel(time=top_da)

    return selected_positions, selected_times, cutoff, top


def generate_subset(
    file_path: Path,
    threshold: float = 0.5,
    quantile: float = 0.75,
    max_frames: int = 1000,
):
    """Generate a subset of frames from a pose estimation file based on confidence scores.
    Parameters
    ----------
    file_path : Path
        Path to the input pose estimation file (NetCDF format).
    threshold : float, optional
        Confidence threshold per keypoint (default is 0.5).
    quantile : float, optional
        Quantile (0-1) of score distribution to use as cutoff (default is 0.75).
    max_frames : int, optional
        Maximum number of frames to return (default is 1000).
    Returns
    -------
    """
    # Load poses from file
    poses = xr.load_dataset(file_path)
    sel_pos, sel_times, cutoff, idx = select_top_frames(
        poses, threshold, quantile, max_frames
    )
    subset = poses.sel(time=sel_times)["position"]
    return torch.tensor(subset.values)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="Path to input pose estimation file (NetCDF format)"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    assert input_path.exists(), f"Input path {input_path} does not exist."
    subset = generate_subset(input_path)
    print(f"Subset shape: {subset.shape}")
