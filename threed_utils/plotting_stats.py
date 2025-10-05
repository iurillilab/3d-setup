import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_speed_subplots(
    dataset: xr.Dataset,
    keypoint_cols: str | Iterable[str] | None = None,
    max_frame: int = 1000,
    ncols: int = 3,
    figsize: tuple[float, float] = (12, 8),
    sharex: bool = True,
    sharey: bool = False,
    save_path: str | None = None,
):
    """
    Make a subplot for each keypoint showing its per-frame speed.

    Speed_k[t] = || position[t+1, :, k] - position[t, :, k] ||_2

    Args:
        dataset: xarray Dataset with 'position' shaped (time, space, keypoints, individuals).
        keypoint_cols: str, iterable of str, or None (plot all keypoints).
        max_frame: slice frames [0:max_frame].
        ncols: number of columns in the subplot grid.
        figsize: overall figure size.
        sharex, sharey: share axes across subplots.
        save_path: if provided, saves the figure.
    Returns:
        (fig, axes)
    """
    assert len(dataset.position.shape) == 4, (
        "Expected (time, space, keypoints, individuals)."
    )

    # --- choose keypoints ---
    if keypoint_cols is None:
        selected_kps = list(dataset.keypoints.values)
    elif isinstance(keypoint_cols, str):
        selected_kps = [keypoint_cols]
    else:
        selected_kps = list(keypoint_cols)

    # --- slice and extract positions ---
    ds_sel = dataset.sel(time=slice(0, max_frame), keypoints=selected_kps)
    pos = ds_sel["position"]
    if "individuals" in pos.dims and pos.sizes.get("individuals", 1) == 1:
        pos = pos.squeeze("individuals")  # (time, space, keypoints)

    arr = pos.values  # (T, S, K)
    darr = np.diff(arr, axis=0)  # (T-1, S, K)
    speed = np.linalg.norm(darr, axis=1)  # (T-1, K) or (T-1,) if K==1

    # time coordinate for the diff'ed series
    t = ds_sel["time"].values
    t = t[1:] if t.size > 1 else t

    # ensure 2D for uniform plotting
    if speed.ndim == 1:
        speed = speed[:, None]

    K = speed.shape[1]
    nrows = math.ceil(K / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    for i, name in enumerate(selected_kps):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.plot(t, speed[:, i])
        ax.set_title(name)
        if r == nrows - 1:
            ax.set_xlabel("Frame")
        if c == 0:
            ax.set_ylabel("Speed (units/frame)")
        ax.grid(True, alpha=0.3)
    # TODO change unit into actuap speed, check fps of the video
    # hide any unused axes
    for j in range(K, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Keypoint speeds over time", y=0.995)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()
    return fig, axes
