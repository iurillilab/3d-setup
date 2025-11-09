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
