"""Module with some utility functions for denoising and cleanign"""

import pickle
from pathlib import Path

import numpy as np
import xarray as xr


def filter_confidence_and_interpolate(
    data: xr.Dataset,
    threshold: float = 0.2,
    max_consecutive: int = 10,
    method: str = None,
    order: int = None,
) -> xr.Dataset:
    """
    Filter a xr.Dataset based on confidence and interpolate missing values.

    Args:
        data: xr.Dataset containing keypoint data
        threshold: float threshold for confidence
        max_consecutive: int maximum number of consecutive missing values to interpolate
        method: (optional) interpolation method to use (e.g., 'linear', 'polynomial', etc.)
        order: (optional) Order of interpolation (if relevant, e.g. for 'polynomial')
    Returns:
        xr.Dataset: filtered and interpolated dataset
    """
    session = data.copy()
    mask = data.confidence.values < threshold
    expanded_mask = mask[:, np.newaxis, :, :]
    session.position.values = np.where(expanded_mask, np.nan, session.position.values)

    interp_kwargs = {
        "dim": "time",
        "max_gap": max_consecutive,
    }
    if method is not None:
        interp_kwargs["method"] = method
    if order is not None:
        interp_kwargs["order"] = order

    session.position.interpolate_na(**interp_kwargs)
    return session


def apply_savgol_filter(
    data: xr.Dataset, window_length: int = 5, polyorder: int = 3
) -> xr.Dataset:
    """
    Apply a Savitzky-Golay filter to the data.
    """
    try:
        import xrscipy.signal as dsp
    except ImportError:
        raise ImportError(
            "xrscipy is not installed. Please install it using `pip install xrscipy`."
        )
    session = data.copy()
    session.position.values = dsp.savgol_filter(
        session.position, window_length=window_length, polyorder=polyorder, dim="time"
    )
    return session


def arena_filtering(
    data: xr.Dataset,
    arena_3d: xr.Dataset,
    tolerance: float = 0.0,
) -> xr.Dataset:
    """
    Filter the 3D coordinates of the animal to be inside the arena.

    Args:
        data: xr.Dataset containing a variable 'position' with dims like (frame, space, point, rep)
              where 'space' has coordinates ['x', 'y', 'z'].
        arena_3d: xr.Dataset containing 'position' with the same 'space' dimension (['x', 'y', 'z']).
        tolerance: float, margin added to the arena boundaries (in same units as position).

    Returns:
        xr.Dataset: copy of input with positions outside the arena set to NaN.
    """
    session = data.copy()
    pos = session["position"]

    # --- Get arena boundaries for each axis with tolerance ---
    x_min = arena_3d.position.sel(space="x").min() - tolerance
    x_max = arena_3d.position.sel(space="x").max() + tolerance
    y_min = arena_3d.position.sel(space="y").min() - tolerance
    y_max = arena_3d.position.sel(space="y").max() + tolerance
    z_min = arena_3d.position.sel(space="z").min() - tolerance
    z_max = arena_3d.position.sel(space="z").max() + tolerance

    # --- Extract each axis from position ---
    x = pos.sel(space="x")
    y = pos.sel(space="y")
    z = pos.sel(space="z")

    # --- Compute mask of points inside the arena ---
    inside = (
        (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
        & (z >= z_min)
        & (z <= z_max)
    )

    # --- Apply mask (xarray broadcasts it over 'space') ---
    session["position"] = pos.where(inside, np.nan)

    return session


def normalize_ax(v, axis=1, eps=1e-8):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(norm, eps, None)


def convert_session_to_egocentric(session: xr.Dataset):
    session_copy = session.copy()
    trunk_names = [
        "back_caudal",
        "back_mid",
        "back_rostral",
        "belly_caudal",
        "belly_rostral",
    ]
    origin = session.position.sel(keypoints=trunk_names).mean(dim="keypoints")
    # define the body axes:
    nose = session.position.sel(keypoints="nose").values
    tailbase = session.position.sel(keypoints="tailbase").values
    ear_lf = session.position.sel(keypoints="ear_lf").values
    ear_rt = session.position.sel(keypoints="ear_rt").values
    origin_np = origin.values
    pose_np = session.position.values

    T, C, K, A = pose_np.shape

    # forward ax
    forward = nose - tailbase
    forward = normalize_ax(forward, axis=1)
    # left ax
    left = ear_lf - ear_rt
    left = normalize_ax(left, axis=1)
    # up ax
    f = forward[..., 0]
    l = left[..., 0]
    # up= forwar X left (right hand rule)
    up = np.cross(f, l, axis=1)
    up = normalize_ax(up, axis=1)
    # recompute hte left for orthogonality
    left_2 = np.cross(up, f, axis=1)
    left_2 = normalize_ax(left_2, axis=1)

    ex = f
    ey = left_2
    ez = up

    # build the rotation matrices
    R = np.stack([ex, ey, ez], axis=2)
    R_T = np.transpose(R, (0, 2, 1))

    # transfrom the keypoints to egocentric
    centered = pose_np - origin_np[:, :, None, :]  # broadcast over kp
    centered = centered[..., 0]
    pose_ego = np.einsum(
        "tij, tjk -> tik", R_T, centered
    )  # we rotate each frame p_body[t] = R_T[t] @ centered[t]
    session_copy["position"].values = np.expand_dims(pose_ego, axis=-1)
    return session_copy
