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
from threed_utils.io import read_calibration_toml
from threed_utils.anipose.triangulate import CameraGroup, triangulate_core


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
    arena_data = np.stack(arena_arrays, axis=0).T
    arena_data = arena_data[np.newaxis, :, :, np.newaxis]
    print(arena_data.shape)
    
    # Create keypoint names (arena corners)
    n_keypoints = arena_data.shape[1]
    keypoint_names = [f"arena_corner_{i}" for i in range(n_keypoints)]
    
    # Create dataset
    ds = xr.Dataset(
        {
            'position': xr.DataArray(
                arena_data,
                dims=['view', 'space','keypoints'],
                coords={
                    'view': cam_names,
                    'keypoints': keypoint_names,
                    'space': ['x', 'y'],
                    'individuals': ['arena']
                }
            ),
            'confidence': xr.DataArray(
                np.ones((len(cam_names), 1, n_keypoints)),
                dims=['view', 'keypoints', 'individuals'],
                coords={
                    'view': cam_names,
                    'keypoints': keypoint_names,
                    'individuals': ['arena']
                }
            )
        }
    )
    
    ds.attrs['source_software'] = 'arena_coordinates'
    ds.attrs['individuals'] = ['arena']
    
    return ds


def triangulate_arena(arena_coordinates: dict, calib_toml_path: Path, 
                     cam_names: list = None) -> xr.Dataset:
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
    if cam_names is None:
        cam_names, _, _, _ = read_calibration_toml(calib_toml_path)
    
    # Create arena dataset
    print(arena_coordinates)
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
    
    # Add time dimension if not present
    if len(positions.shape) == 3:
        positions = positions[np.newaxis, ...]  # (1, n_views, n_keypoints, 2)
        scores = scores[np.newaxis, ...]       # (1, n_views, n_keypoints)
    
    # Triangulate
    triang_df = triangulate_core(
        config, 
        positions, 
        scores, 
        arena_2d_ds.coords["keypoints"].values, 
        cgroup, 
    )
    
    # Convert to movement dataset
    arena_3d_ds = movement_ds_from_anipose_triangulation_df(triang_df, individual_name="arena")
    
    return arena_3d_ds


def movement_ds_from_anipose_triangulation_df(triang_df, individual_name="arena"):
    """Convert triangulation dataframe to xarray dataset for arena points."""
    keypoint_names = sorted(list(set([col.rsplit('_', 1)[0] for col in triang_df.columns 
                                   if any(col.endswith(f'_{s}') for s in ['x','y','z'])])))

    n_frames = len(triang_df)
    n_keypoints = len(keypoint_names)

    # Initialize arrays and fill
    position_array = np.zeros((n_frames, 1, n_keypoints, 3))  # 1 for single individual
    confidence_array = np.zeros((n_frames, 1, n_keypoints))
    
    for i, kp in enumerate(keypoint_names):
        for j, coord in enumerate(['x', 'y', 'z']):
            position_array[:, 0, i, j] = triang_df[f'{kp}_{coord}']
        confidence_array[:, 0, i] = triang_df[f'{kp}_score']

    individual_names = [individual_name]
    position_array = position_array.transpose(0, 3, 2, 1)
    confidence_array = confidence_array.transpose(0, 2, 1)

    return from_numpy(position_array=position_array,
                     confidence_array=confidence_array, 
                     individual_names=individual_names,
                     keypoint_names=keypoint_names,
                     source_software="anipose_triangulation")


def get_arena_points_from_dataset(arena_ds: xr.Dataset, time_idx: int = 0) -> np.ndarray:
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
    positions = arena_ds.position.isel(time=time_idx, individuals=0)
    arena_points = positions.transpose('keypoints', 'space').values
    return arena_points


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