"""Script to apply Savitzky-Golay filter to a triangulations."""

from argparse import ArgumentParser
from os import wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pandas.errors import ParserError
from plotting_stats import plot_speed_subplots
from scipy.signal import butter, filtfilt, savgol_filter
from tqdm import tqdm


def denoise_positions_savgol(ds: xr.Dataset, window_s=0.1, polyorder=2):
    fps = 60
    window_length = int(window_s * fps)
    w = max(3, int(round(window_s * fps)))
    if w % 2 == 0:
        w += 1  # window length must be odd

    pos = ds["position"]
    arr = pos.values  # (T, S, K, I)
    arr_sm = savgol_filter(
        arr, window_length=w, polyorder=polyorder, axis=0, mode="interp"
    )
    return pos.copy(data=arr_sm)


def denoise_position_lowpass(ds: xr.Dataset, fc_hz=6.0, order=4):
    fps = 60
    wn = fc_hz / (fps / 2.0)  # Normalize the frequency
    b, a = butter(order, wn, btype="lowpass")
    pos = ds["position"].values  # (T, S, K, I)
    arr_sm = filtfilt(b, a, pos, axis=0, padlen=3 * max(len(a), len(b)) - 1)
    return ds["position"].copy(data=arr_sm)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input  file containing the triangulation data.",
    )
    PARSER.add_argument("-k", "--keypoints", nargs="+", action="extend", default=None)
    PARSER.add_argument("--filter", type=str, default="savgol")
    args = PARSER.parse_args()
    INPUT_PATH = Path(args.input_path)
    filtering_method = args.filter
    data_original = xr.open_dataset(INPUT_PATH)
    data = data_original.copy()
    if args.keypoints is not None:
        if len(args.keypoints) == 1:
            data = data.sel(keypoints=args.keypoints[0])
        else:
            data = data.sel(keypoints=args.keypoints)
    if filtering_method == "savgol":
        data["position"] = denoise_positions_savgol(data)
    elif filtering_method == "lowpass":
        data["position"] = denoise_position_lowpass(data)
    # produce the two plots
    plot_speed_subplots(
        data_original,
        keypoint_cols=args.keypoints,
        max_frame=1000,
        ncols=4,
        figsize=(16, 10),
        title="Original Data",
    )
    data["position"] = data.position.values.reshape(data.position.shape + (1,))
    plot_speed_subplots(
        data,
        keypoint_cols=args.keypoints,
        max_frame=1000,
        ncols=4,
        figsize=(16, 10),
        title=f"Filtered Data ({filtering_method})",
    )
