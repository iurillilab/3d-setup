"""
Arena triangulation and visualization utilities.

This module provides functions to triangulate arena points from 2D coordinates
and visualize them together with mouse tracking data.
"""

import json
import numpy as np
import xarray as xr
from pathlib import Path
from movement.io.load_poses import from_numpy
from threed_utils.io import read_calibration_toml, movement_ds_from_anipose_triangulation_df
from threed_utils.anipose.triangulate import CameraGroup, triangulate_core
from threed_utils.anipose.movement_anipose import anipose_triangulate_ds


def load_arena_coordinates(arena_json_path: Path) -> dict:
    """
    Load arena coordinates from JSON file.
    
    Parameters
    ----------
    arena_json_path : Path
        Path to the arena JSON file
        
    Returns
    -------
    dict
        Dictionary containing arena coordinates for each camera view
    """
    with open(arena_json_path, 'r') as f:
        data = json.load(f)
    
    # Get the last entry which contains the points_coordinate
    arena_data = data[-1]['points_coordinate']
    return arena_data


def create_arena_dataset(arena_coordinates: dict, cam_names: list) -> xr.Dataset:
    """
    Create a movement dataset from arena coordinates.
    
    Parameters
    ----------
    arena_coordinates : dict
        Dictionary containing arena coordinates for each camera view
    cam_names : list
        List of camera names in the order they should be used
        
    Returns
    -------
    xarray.Dataset
        Dataset containing arena coordinates with dimensions (view, keypoints, space)
    """
    # Convert arena coordinates to arrays
    arena_arrays = []
    for cam_name in cam_names:
        if cam_name in arena_coordinates:
            coords = np.array(arena_coordinates[cam_name])
            arena_arrays.append(coords)
        else:
            raise ValueError(f"Camera {cam_name} not found in arena coordinates")
    
    # Stack arrays to create (1, 2, n_keypoints, 1) array
    arena_data = np.stack(arena_arrays, axis=0).swapaxes(1, 2)[:, np.newaxis, :, :, np.newaxis]

    # Create keypoint names (arena corners)
    n_keypoints = 8
    keypoint_names = [f"arena_corner_{i}" for i in range(n_keypoints)]
    
    # Create dataset
    print(keypoint_names)
    print(arena_data.shape)
    print(cam_names)
    ds = xr.Dataset(
        {
            'position': xr.DataArray(
                arena_data,
                dims=['view', 'time', 'space', 'keypoints', 'individuals'],
                coords={
                    'view': cam_names,
                    'time': [0],
                    'space': ['x', 'y'],
                    'keypoints': keypoint_names,
                    'individuals': ['arena']
                }
            ),
            'confidence': xr.DataArray(
                np.ones_like(arena_data)[:, :, 0, :, :],
                dims=['view', 'time', 'keypoints', 'individuals'],
                coords={
                    'view': cam_names,
                    'time': [0],
                    'keypoints': keypoint_names,
                    'individuals': ['arena']
                }
            )
        }
    )
    
    ds.attrs['source_software'] = 'arena_coordinates'
    ds.attrs['individuals'] = ['arena']
    
    return ds


def triangulate_arena(arena_coordinates: dict, calib_toml_path: Path) -> xr.Dataset:
    """
    Triangulate arena points using calibration data.
    
    Parameters
    ----------
    arena_coordinates : dict
        Dictionary containing arena coordinates for each camera view
    calib_toml_path : Path
        Path to calibration TOML file
    cam_names : list, optional
        List of camera names. If None, will be read from calibration file
        
    Returns
    -------
    xarray.Dataset
        Dataset containing triangulated 3D arena points
    """
    # Load calibration data
    cam_names, _, _, _ = read_calibration_toml(calib_toml_path)
    
    # Create arena dataset
    arena_2d_ds = create_arena_dataset(arena_coordinates, cam_names)
    print(arena_2d_ds, arena_2d_ds.position.shape)
    
    # Triangulate using anipose
    triang_config = {
        "ransac": True,
        "optim": False,
        "score_threshold": 0.0,
        "reproj_error_threshold": 150,
    }
    
    config = dict(triangulation=triang_config)
    
    calib_fname = str(calib_toml_path)
    cgroup = CameraGroup.load(calib_fname)
    
    # Prepare data for triangulation
    positions = arena_2d_ds.position.values  # (n_views, n_keypoints, 2)
    scores = arena_2d_ds.confidence.values   # (n_views, n_keypoints)
    
    # Triangulate
    arena_3d_ds = anipose_triangulate_ds(
        arena_2d_ds, 
        calib_toml_path, 
        **triang_config, 
    )

    return arena_3d_ds


# def get_arena_points_from_dataset(arena_ds: xr.Dataset, time_idx: int = 0) -> np.ndarray:
def get_triangulated_arena_ds(arena_points_json_path: Path, calib_toml_path: Path) -> xr.Dataset:
    """
    Extract arena points from dataset for plotting.
    
    Parameters
    ----------
    arena_ds : xarray.Dataset
        Dataset containing triangulated arena points
    time_idx : int, optional
        Time index to extract (default: 0)
        
    Returns
    -------
    np.ndarray
        Array of arena points with shape (n_points, 3)
    """
    arena_coordinates = load_arena_coordinates(arena_points_json_path)
    # arena_ds = create_arena_dataset(arena_coordinates, calib_toml_path)
    arena_3d_ds = triangulate_arena(arena_coordinates, calib_toml_path)
    return arena_3d_ds


def plot_arena_with_mouse(mouse_ds: xr.Dataset, arena_ds: xr.Dataset, 
                         time_idx: int = 0, individual_idx: int = 0,
                         figsize: tuple = (12, 10), save_path: Path = None):
    """
    Plot arena and mouse together in 3D.
    
    Parameters
    ----------
    mouse_ds : xarray.Dataset
        Dataset containing mouse tracking data
    arena_ds : xarray.Dataset
        Dataset containing arena points
    time_idx : int, optional
        Time index to plot (default: 0)
    individual_idx : int, optional
        Individual index to plot (default: 0)
    figsize : tuple, optional
        Figure size (default: (12, 10))
    save_path : Path, optional
        Path to save the plot (default: None)
    """
    import matplotlib.pyplot as plt
    from threed_utils.visualization.skeleton_plots import plot_skeleton_3d, set_axes_equal
    
    # Get arena points
    arena_points = get_arena_points_from_dataset(arena_ds, time_idx)
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot arena points
    ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
              c='lightgray', s=50, alpha=0.8, label='Arena')
    
    # Plot mouse skeleton
    plot_skeleton_3d(mouse_ds, time_idx=time_idx, individual_idx=individual_idx,
                    ax=ax, arena_points=None)  # Don't plot arena again
    
    ax.set_title(f'Arena and Mouse - Frame {time_idx}')
    
    # Apply equal axes scaling
    set_axes_equal(ax)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax


def get_arena_points_from_dataset(arena_ds: xr.Dataset, time_idx: int = 0) -> np.ndarray:
    """
    Extract arena points from dataset for plotting.
    
    Parameters
    ----------
    arena_ds : xarray.Dataset   

    Returns
    -------
    np.ndarray
        Array of arena points with shape (n_points, 3)
    """
    arena_points = arena_ds.position.isel(time=time_idx, individuals=0)
    return arena_points.transpose('keypoints', 'space').values


def create_arena_mouse_animation(mouse_ds: xr.Dataset, arena_ds: xr.Dataset,
                               start_time: int = 0, end_time: int = None,
                               individual_idx: int = 0, interval: int = 100,
                               save_path: Path = None):
    """
    Create animation showing arena and mouse movement.
    
    Parameters
    ----------
    mouse_ds : xarray.Dataset
        Dataset containing mouse tracking data
    arena_ds : xarray.Dataset
        Dataset containing arena points
    start_time : int, optional
        Start time index (default: 0)
    end_time : int, optional
        End time index (default: None, uses all available time)
    individual_idx : int, optional
        Individual index to plot (default: 0)
    interval : int, optional
        Animation interval in milliseconds (default: 100)
    save_path : Path, optional
        Path to save the animation (default: None)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from threed_utils.visualization.skeleton_plots import plot_skeleton_3d, set_axes_equal
    
    if end_time is None:
        end_time = mouse_ds.sizes['time']
    
    # Get arena points
    arena_points = get_arena_points_from_dataset(arena_ds, 0)  # Arena is static
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits based on data
    positions = mouse_ds.position.isel(individuals=individual_idx).sel(time=slice(start_time, end_time))
    x_data = positions.sel(space='x').values
    y_data = positions.sel(space='y').values
    z_data = positions.sel(space='z').values
    
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(z_data))
    if np.any(valid_mask):
        ax.set_xlim(x_data[valid_mask].min(), x_data[valid_mask].max())
        ax.set_ylim(y_data[valid_mask].min(), y_data[valid_mask].max())
        ax.set_zlim(z_data[valid_mask].min(), z_data[valid_mask].max())
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_box_aspect([1, 1, 1])
    
    def animate(frame_idx):
        ax.clear()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Plot arena points
        ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
                  c='lightgray', s=50, alpha=0.8, label='Arena')
        
        if np.any(valid_mask):
            ax.set_xlim(x_data[valid_mask].min(), x_data[valid_mask].max())
            ax.set_ylim(y_data[valid_mask].min(), y_data[valid_mask].max())
            ax.set_zlim(z_data[valid_mask].min(), z_data[valid_mask].max())
        
        ax.set_box_aspect([1, 1, 1])
        
        # Plot mouse skeleton
        plot_skeleton_3d(mouse_ds, time_idx=start_time + frame_idx, individual_idx=individual_idx,
                        ax=ax, arena_points=None)
        ax.set_title(f'Arena and Mouse - Frame {start_time + frame_idx}')
        
        set_axes_equal(ax)
    
    anim = FuncAnimation(fig, animate, frames=end_time-start_time, 
                       interval=interval, blit=False, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=30)
    
    return anim 