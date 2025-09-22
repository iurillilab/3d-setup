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

    

def load_arena_multiview_ds(arena_json_path: Path) -> xr.Dataset:
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
    arena_coordinates = load_arena_coordinates(arena_json_path)

    # Convert arena coordinates to arrays
    view_names = sorted(arena_coordinates.keys())
    arena_arrays = []
    for view_name in view_names:
        coords = np.array(arena_coordinates[view_name])
        arena_arrays.append(coords)

    # CAREFUL! Some arbitrary shaping of the array is happening at this level, to ensure 
    # that loaded stuff is interpreted correctly
    # Stack arrays to create (1, 2, n_keypoints, 1) array
    arena_data = np.stack(arena_arrays, axis=0).swapaxes(1, 2)[:, np.newaxis, :, :, np.newaxis][:, :, [1, 0], :, :]

    # Create keypoint names (arena corners)
    n_keypoints = 8
    keypoint_names = [f"arena_corner_{i}" for i in range(n_keypoints)]
    
    # Create dataset
    ds = xr.Dataset(
        {
            'position': xr.DataArray(
                arena_data,
                dims=['view', 'time', 'space', 'keypoints', 'individuals'],
                coords={
                    'view': view_names,
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
                    'view': view_names,
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


def triangulate_arena(arena_multiview_ds: dict, calib_toml_path: Path) -> xr.Dataset:
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
    # Triangulate using anipose
    triang_config = {
        "ransac": True,
        "optim": False,
        "score_threshold": 0.0,
        "reproj_error_threshold": 150,
    }    
    
    # Triangulate
    arena_3d_ds = anipose_triangulate_ds(
        arena_multiview_ds, 
        calib_toml_path, 
        **triang_config, 
    )

    return arena_3d_ds


# def get_arena_points_from_dataset(arena_ds: xr.Dataset, time_idx: int = 0) -> np.ndarray:
def load_and_triangulate_arena_ds(arena_points_json_path: Path, calib_toml_path: Path) -> xr.Dataset:
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
    arena_multiview_ds = load_arena_multiview_ds(arena_points_json_path)
    # arena_ds = create_arena_dataset(arena_coordinates, calib_toml_path)
    arena_3d_ds = triangulate_arena(arena_multiview_ds, calib_toml_path)
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


def apply_affine_transformation(ds: xr.Dataset, transformation_matrix: np.ndarray) -> xr.Dataset:
    """
    Apply a 3D affine transformation to a triangulated dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing triangulated 3D points with dimensions (time, space, keypoints, individuals)
    transformation_matrix : np.ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    xarray.Dataset
        Transformed dataset with the same structure as input
    """
    # Extract positions and convert to homogeneous coordinates
    positions = ds.position.values  # (time, space, keypoints, individuals)
    
    # Reshape to (n_points, 4) for homogeneous coordinates
    n_time, n_space, n_keypoints, n_individuals = positions.shape
    n_points = n_time * n_keypoints * n_individuals
    
    # Stack all points and add homogeneous coordinate
    points_flat = positions.transpose(0, 2, 3, 1).reshape(n_points, 3)
    points_homogeneous = np.column_stack([points_flat, np.ones(n_points)])
    
    # Apply transformation
    transformed_points_homogeneous = points_homogeneous @ transformation_matrix.T
    
    # Remove homogeneous coordinate and reshape back
    transformed_points = transformed_points_homogeneous[:, :3]
    transformed_positions = transformed_points.reshape(n_time, n_keypoints, n_individuals, 3).transpose(0, 3, 1, 2)
    
    # Create new dataset with transformed positions
    transformed_ds = ds.copy()
    transformed_ds['position'] = xr.DataArray(
        transformed_positions,
        dims=['time', 'space', 'keypoints', 'individuals'],
        coords=ds.position.coords
    )
    
    return transformed_ds


def find_orthogonal_affine_transformation(arena_ds: xr.Dataset, 
                                        origin_corner: str = "arena_corner_0",
                                        x_axis_points: tuple = ("arena_corner_0", "arena_corner_1"),
                                        y_axis_points: tuple = ("arena_corner_0", "arena_corner_3"),
                                        z_axis_points: tuple = ("arena_corner_0", "arena_corner_4")) -> np.ndarray:
    """
    Find an affine transformation that makes arena axes orthogonal while preserving volumes.
    
    Parameters
    ----------
    arena_ds : xarray.Dataset
        Dataset containing triangulated arena points
    origin_corner : str, optional
        Name of the corner to use as origin (default: "arena_corner_0")
    x_axis_points : tuple, optional
        Tuple of two corner names defining the x-axis direction (default: ("arena_corner_0", "arena_corner_1"))
    y_axis_points : tuple, optional
        Tuple of two corner names defining the y-axis direction (default: ("arena_corner_0", "arena_corner_3"))
    z_axis_points : tuple, optional
        Tuple of two corner names defining the z-axis direction (default: ("arena_corner_0", "arena_corner_4"))
        
    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix
    """
    # Extract points for each axis - take first time point and first individual
    individual = 'checkerboard'
    origin_point = arena_ds.position.sel(keypoints=origin_corner, time=0, individuals=individual).values
    x_start = arena_ds.position.sel(keypoints=x_axis_points[0], time=0, individuals=individual).values
    x_end = arena_ds.position.sel(keypoints=x_axis_points[1], time=0, individuals=individual).values
    y_start = arena_ds.position.sel(keypoints=y_axis_points[0], time=0, individuals=individual).values
    y_end = arena_ds.position.sel(keypoints=y_axis_points[1], time=0, individuals=individual).values
    z_start = arena_ds.position.sel(keypoints=z_axis_points[0], time=0, individuals=individual).values
    z_end = arena_ds.position.sel(keypoints=z_axis_points[1], time=0, individuals=individual).values
    
    # Calculate direction vectors
    x_dir = x_end - x_start
    y_dir = y_end - y_start
    z_dir = z_end - z_start
    
    # Normalize primary axis (x)
    x_axis = x_dir / np.linalg.norm(x_dir)
    # Orthogonalize y to x
    y_axis = y_dir - np.dot(y_dir, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Compute z as right-handed cross product
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Re-orthogonalize y to z (optional, for numerical stability)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Build orthogonal basis (current arena axes)
    arena_basis = np.column_stack([x_axis, y_axis, z_axis])
    # Target basis is the standard coordinate system
    target_basis = np.eye(3)  # (1,0,0), (0,1,0), (0,0,1)
    # Compute rotation that maps arena basis to target basis
    # R * arena_basis = target_basis, so R = target_basis * arena_basis^(-1)
    R = target_basis @ np.linalg.inv(arena_basis)
    # Scale to preserve volume
    current_volume = np.abs(np.linalg.det(arena_basis))
    target_volume = np.abs(np.linalg.det(target_basis))
    scale_factor = (current_volume / target_volume) ** (1/3)
    R = R * scale_factor
    # Build affine transformation: rotate to align with standard axes, then translate origin to (0,0,0)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = -R @ origin_point
    return transformation_matrix


def find_optimal_affine_transformation(arena_ds: xr.Dataset, 
                                     origin_corner: str = "arena_corner_0",
                                     x_axis_points: tuple = ("arena_corner_0", "arena_corner_1"),
                                     y_axis_points: tuple = ("arena_corner_0", "arena_corner_3"),
                                     z_axis_points: tuple = ("arena_corner_0", "arena_corner_4")) -> np.ndarray:
    """
    Find optimal affine transformation that minimizes angles between arena axes and standard coordinate axes.
    Uses least squares to find the best affine fit, allowing non-rigid transformations.
    Hard constraint: xy-plane (corners 0,1,2,3) must be flat and aligned with standard xy-plane (z=0).
    
    Parameters
    ----------
    arena_ds : xarray.Dataset
        Dataset containing triangulated arena points
    origin_corner : str, optional
        Name of the corner to use as origin (default: "arena_corner_0")
    x_axis_points : tuple, optional
        Tuple of two corner names defining the x-axis direction (default: ("arena_corner_0", "arena_corner_1"))
    y_axis_points : tuple, optional
        Tuple of two corner names defining the y-axis direction (default: ("arena_corner_0", "arena_corner_3"))
    z_axis_points : tuple, optional
        Tuple of two corner names defining the z-axis direction (default: ("arena_corner_0", "arena_corner_4"))
        
    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix
    """
    # Extract all arena points for the first time and individual
    individual = 'checkerboard'
    origin_point = arena_ds.position.sel(keypoints=origin_corner, time=0, individuals=individual).values
    
    # Get all arena points
    all_points = []
    
    # Extract all keypoints
    for i in range(8):  # 8 arena corners
        keypoint_name = f"arena_corner_{i}"
        point = arena_ds.position.sel(keypoints=keypoint_name, time=0, individuals=individual).values
        all_points.append(point)
    
    all_points = np.array(all_points)  # Shape: (8, 3)
    
    # Define target positions with xy-plane constraint
    # xy-plane points (corners 0,1,2,3) must have z=0
    target_points = np.array([
        [0, 0, 0],    # origin (arena_corner_0) - xy-plane
        [1, 0, 0],    # x-axis (arena_corner_1) - xy-plane
        [1, 1, 0],    # xy-plane (arena_corner_2) - xy-plane
        [0, 1, 0],    # y-axis (arena_corner_3) - xy-plane
        [0, 0, 1],    # z-axis (arena_corner_4) - above xy-plane
        [1, 0, 1],    # xz-plane (arena_corner_5) - above xy-plane
        [1, 1, 1],    # xyz (arena_corner_6) - above xy-plane
        [0, 1, 1],    # yz-plane (arena_corner_7) - above xy-plane
    ])
    
    # Scale target points to match the scale of arena points
    arena_scale = np.max(np.linalg.norm(all_points - all_points[0], axis=1))
    target_scale = np.max(np.linalg.norm(target_points - target_points[0], axis=1))
    scale_factor = arena_scale / target_scale
    target_points = target_points * scale_factor
    
    # Solve for affine transformation using least squares with xy-plane constraint
    # We need to solve: A * [points; 1] = target_points
    # But we also need to ensure that xy-plane points (0,1,2,3) have z=0
    
    # Add homogeneous coordinate
    points_homogeneous = np.column_stack([all_points, np.ones(len(all_points))])
    
    # Solve least squares: A * points_homogeneous.T = target_points.T
    # A = target_points.T @ points_homogeneous @ inv(points_homogeneous.T @ points_homogeneous)
    A = target_points.T @ points_homogeneous @ np.linalg.inv(points_homogeneous.T @ points_homogeneous)
    
    # A is 3x4, we need 4x4 affine matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :] = A
    
    # Apply the transformation and verify xy-plane constraint
    transformed_points = apply_affine_transformation_to_points(all_points, transformation_matrix)
    
    # Check if xy-plane points (0,1,2,3) have z≈0
    xy_plane_indices = [0, 1, 2, 3]
    xy_plane_z_values = transformed_points[xy_plane_indices, 2]
    print(f"XY-plane z-values after transformation: {xy_plane_z_values}")
    
    # If xy-plane is not flat, adjust the transformation
    if np.any(np.abs(xy_plane_z_values) > 1e-6):
        print("Adjusting transformation to ensure xy-plane is flat...")
        # Force xy-plane points to have z=0 by adjusting the transformation
        # This is a simple approach: set the z-component of xy-plane points to 0
        for idx in xy_plane_indices:
            target_points[idx, 2] = 0
        
        # Recompute transformation
        A = target_points.T @ points_homogeneous @ np.linalg.inv(points_homogeneous.T @ points_homogeneous)
        transformation_matrix[:3, :] = A
    
    return transformation_matrix


def apply_affine_transformation_to_points(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        Points to transform, shape (n_points, 3)
    transformation_matrix : np.ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    np.ndarray
        Transformed points, shape (n_points, 3)
    """
    # Add homogeneous coordinate
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_points_homogeneous = points_homogeneous @ transformation_matrix.T
    
    # Remove homogeneous coordinate
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points


if __name__ == "__main__":
    arena_json_path = Path("/Users/vigji/Desktop/test_3d/multicam_video_2025-05-07T10_12_11_20250528-153946.json")
    calib_toml_path = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328/mc_calibration_output_20250710-152443/calibration_from_mc.toml")
    
    arena_multiview_ds = load_arena_multiview_ds(arena_json_path)
    arena_ds = load_and_triangulate_arena_ds(arena_json_path, calib_toml_path)

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Plot original arena
    fig = plt.figure(figsize=(15, 5))
    
    # Original arena
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(arena_ds.position.sel(space='x').values, arena_ds.position.sel(space='y').values, arena_ds.position.sel(space='z').values, c='lightgray', s=5, alpha=0.8, label='Arena')
    
    # Define axis edges
    ax_edges = dict(x=["arena_corner_0", "arena_corner_1"], 
                    y=["arena_corner_0", "arena_corner_3"], 
                    z=["arena_corner_0", "arena_corner_4"],
                    x1=["arena_corner_0", "arena_corner_2"],
                    x2=["arena_corner_1", "arena_corner_3"],
                    x3=["arena_corner_1", "arena_corner_2"],
                    x4=["arena_corner_3", "arena_corner_2"])

    for ax_col, (ax_name, ax_edge) in zip(["r", "g", "b", "y"]*2, ax_edges.items()):
        sel_arena_corner = arena_ds.position.sel(keypoints=ax_edge[0])
        sel_arena_corner_2 = arena_ds.position.sel(keypoints=ax_edge[1])
        ax1.plot([sel_arena_corner.sel(space="x").values, sel_arena_corner_2.sel(space="x").values], 
                [sel_arena_corner.sel(space="y").values, sel_arena_corner_2.sel(space="y").values], 
                [sel_arena_corner.sel(space="z").values, sel_arena_corner_2.sel(space="z").values], 
                c=ax_col, label=f"Arena edge {ax_name}")
    ax1.legend()
    ax1.set_title("Original Arena")
    
    # Find orthogonal transformation
    transformation_matrix = find_orthogonal_affine_transformation(arena_ds)
    
    # Apply transformation
    transformed_arena_ds = apply_affine_transformation(arena_ds, transformation_matrix)
    
    # Find optimal affine transformation (non-rigid)
    optimal_transformation_matrix = find_optimal_affine_transformation(arena_ds)
    
    # Apply optimal transformation
    optimal_transformed_arena_ds = apply_affine_transformation(arena_ds, optimal_transformation_matrix)
    
    # Plot transformed arena (orthogonal)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(transformed_arena_ds.position.sel(space='x').values, transformed_arena_ds.position.sel(space='y').values, transformed_arena_ds.position.sel(space='z').values, c='lightgray', s=5, alpha=0.8, label='Orthogonalized Arena')
    
    for ax_col, (ax_name, ax_edge) in zip(["r", "g", "b"], ax_edges.items()):
        sel_arena_corner = transformed_arena_ds.position.sel(keypoints=ax_edge[0])
        sel_arena_corner_2 = transformed_arena_ds.position.sel(keypoints=ax_edge[1])
        ax2.plot([sel_arena_corner.sel(space="x").values, sel_arena_corner_2.sel(space="x").values], 
                [sel_arena_corner.sel(space="y").values, sel_arena_corner_2.sel(space="y").values], 
                [sel_arena_corner.sel(space="z").values, sel_arena_corner_2.sel(space="z").values], 
                c=ax_col, label=f"Arena edge {ax_name}")
    ax2.legend()
    ax2.set_title("Orthogonalized Arena")
    
    # Plot optimal transformed arena (non-rigid)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(optimal_transformed_arena_ds.position.sel(space='x').values, optimal_transformed_arena_ds.position.sel(space='y').values, optimal_transformed_arena_ds.position.sel(space='z').values, c='lightgray', s=5, alpha=0.8, label='Optimal Transformed Arena')
    
    for ax_col, (ax_name, ax_edge) in zip(["r", "g", "b"], ax_edges.items()):
        sel_arena_corner = optimal_transformed_arena_ds.position.sel(keypoints=ax_edge[0])
        sel_arena_corner_2 = optimal_transformed_arena_ds.position.sel(keypoints=ax_edge[1])
        ax3.plot([sel_arena_corner.sel(space="x").values, sel_arena_corner_2.sel(space="x").values], 
                [sel_arena_corner.sel(space="y").values, sel_arena_corner_2.sel(space="y").values], 
                [sel_arena_corner.sel(space="z").values, sel_arena_corner_2.sel(space="z").values], 
                c=ax_col, label=f"Arena edge {ax_name}")
    ax3.legend()
    ax3.set_title("Optimal Affine Transform")
    
    plt.tight_layout()
    plt.show()
    
    # Print transformation matrices
    print("Orthogonal transformation matrix:")
    print(transformation_matrix)
    print("\nOptimal transformation matrix:")
    print(optimal_transformation_matrix)
    
    # Verify orthogonality for both transformations
    print("\nVerifying orthogonality:")
    x_dir_orig = arena_ds.position.sel(keypoints="arena_corner_1").values - arena_ds.position.sel(keypoints="arena_corner_0").values
    y_dir_orig = arena_ds.position.sel(keypoints="arena_corner_3").values - arena_ds.position.sel(keypoints="arena_corner_0").values
    z_dir_orig = arena_ds.position.sel(keypoints="arena_corner_4").values - arena_ds.position.sel(keypoints="arena_corner_0").values
    
    x_dir_trans = transformed_arena_ds.position.sel(keypoints="arena_corner_1").values - transformed_arena_ds.position.sel(keypoints="arena_corner_0").values
    y_dir_trans = transformed_arena_ds.position.sel(keypoints="arena_corner_3").values - transformed_arena_ds.position.sel(keypoints="arena_corner_0").values
    z_dir_trans = transformed_arena_ds.position.sel(keypoints="arena_corner_4").values - transformed_arena_ds.position.sel(keypoints="arena_corner_0").values
    
    x_dir_opt = optimal_transformed_arena_ds.position.sel(keypoints="arena_corner_1").values - optimal_transformed_arena_ds.position.sel(keypoints="arena_corner_0").values
    y_dir_opt = optimal_transformed_arena_ds.position.sel(keypoints="arena_corner_3").values - optimal_transformed_arena_ds.position.sel(keypoints="arena_corner_0").values
    z_dir_opt = optimal_transformed_arena_ds.position.sel(keypoints="arena_corner_4").values - optimal_transformed_arena_ds.position.sel(keypoints="arena_corner_0").values
    
    # Normalize
    x_dir_orig = x_dir_orig / np.linalg.norm(x_dir_orig)
    y_dir_orig = y_dir_orig / np.linalg.norm(y_dir_orig)
    z_dir_orig = z_dir_orig / np.linalg.norm(z_dir_orig)
    x_dir_trans = x_dir_trans / np.linalg.norm(x_dir_trans)
    y_dir_trans = y_dir_trans / np.linalg.norm(y_dir_trans)
    z_dir_trans = z_dir_trans / np.linalg.norm(z_dir_trans)
    x_dir_opt = x_dir_opt / np.linalg.norm(x_dir_opt)
    y_dir_opt = y_dir_opt / np.linalg.norm(y_dir_opt)
    z_dir_opt = z_dir_opt / np.linalg.norm(z_dir_opt)
    
    print(f"Original dot products: x·y={np.dot(x_dir_orig, y_dir_orig):.6f}, x·z={np.dot(x_dir_orig, z_dir_orig):.6f}, y·z={np.dot(y_dir_orig, z_dir_orig):.6f}")
    print(f"Orthogonalized dot products: x·y={np.dot(x_dir_trans, y_dir_trans):.6f}, x·z={np.dot(x_dir_trans, z_dir_trans):.6f}, y·z={np.dot(y_dir_trans, z_dir_trans):.6f}")
    print(f"Optimal affine dot products: x·y={np.dot(x_dir_opt, y_dir_opt):.6f}, x·z={np.dot(x_dir_opt, z_dir_opt):.6f}, y·z={np.dot(y_dir_opt, z_dir_opt):.6f}")
    
    # Check alignment with standard axes
    standard_x = np.array([1, 0, 0])
    standard_y = np.array([0, 1, 0])
    standard_z = np.array([0, 0, 1])
    
    print(f"\nAlignment with standard axes:")
    print(f"Orthogonalized - x alignment: {np.abs(np.dot(x_dir_trans, standard_x)):.6f}")
    print(f"Orthogonalized - y alignment: {np.abs(np.dot(y_dir_trans, standard_y)):.6f}")
    print(f"Orthogonalized - z alignment: {np.abs(np.dot(z_dir_trans, standard_z)):.6f}")
    print(f"Optimal affine - x alignment: {np.abs(np.dot(x_dir_opt, standard_x)):.6f}")
    print(f"Optimal affine - y alignment: {np.abs(np.dot(y_dir_opt, standard_y)):.6f}")
    print(f"Optimal affine - z alignment: {np.abs(np.dot(z_dir_opt, standard_z)):.6f}")