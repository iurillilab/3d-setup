"""
Skeleton plotting utilities for 3D triangulated datasets.

This module provides functions to visualize skeleton data from triangulated datasets,
including single frame plots, trajectory plots, animations, and multi-frame comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray as xr
from pathlib import Path


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_skeleton_3d(dataset, time_idx=0, individual_idx=0, ax=None, 
                    keypoint_size=50, segment_width=2, keypoint_color='red', 
                    segment_color='blue', alpha=0.8, arena_points=None, arena_color='lightgray'):
    """
    Plot the skeleton from a triangulated dataset at a specific time point.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing position data with dimensions (time, individuals, keypoints, space)
    time_idx : int, optional
        Time index to plot (default: 0)
    individual_idx : int, optional
        Individual index to plot (default: 0)
    ax : matplotlib.axes.Axes, optional
        Matplotlib 3D axes to plot on. If None, creates a new figure
    keypoint_size : int, optional
        Size of keypoint markers (default: 50)
    segment_width : int, optional
        Width of skeleton segment lines (default: 2)
    keypoint_color : str, optional
        Color for keypoint markers (default: 'red')
    segment_color : str, optional
        Color for skeleton segments (default: 'blue')
    alpha : float, optional
        Transparency for segments (default: 0.8)
    arena_points : np.ndarray, optional
        Arena points to plot (shape: n_points, 3) (default: None)
    arena_color : str, optional
        Color for arena points (default: 'lightgray')
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot arena points if provided
    if arena_points is not None:
        ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
                  c=arena_color, s=30, alpha=0.6, label='Arena')
    
    # Get position data for the specified time and individual
    positions = dataset.position.isel(time=time_idx, individuals=individual_idx)
    
    # Get keypoint names
    keypoint_names = list(dataset.coords['keypoints'].values)
    
    # Get skeleton from dataset attributes
    skeleton = dataset.attrs.get('skeleton', [])
    
    # Convert skeleton from keypoint names to indices
    keypoint_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    skeleton_indices = []
    for start_kp, end_kp in skeleton:
        if start_kp in keypoint_to_idx and end_kp in keypoint_to_idx:
            skeleton_indices.append((keypoint_to_idx[start_kp], keypoint_to_idx[end_kp]))
    
    # Extract coordinates
    x_coords = positions.sel(space='x').values
    y_coords = positions.sel(space='y').values
    z_coords = positions.sel(space='z').values
    
    # Plot keypoints (larger points)
    valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords) | np.isnan(z_coords))
    if np.any(valid_mask):
        ax.scatter(x_coords[valid_mask], y_coords[valid_mask], z_coords[valid_mask], 
                  s=keypoint_size, c=keypoint_color, alpha=alpha, label='Keypoints')
    
    # Plot skeleton segments
    for start_idx, end_idx in skeleton_indices:
        # Get coordinates for this segment
        start_x, start_y, start_z = x_coords[start_idx], y_coords[start_idx], z_coords[start_idx]
        end_x, end_y, end_z = x_coords[end_idx], y_coords[end_idx], z_coords[end_idx]
        
        # Check if both endpoints are valid (not NaN)
        if not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(start_z) or
                np.isnan(end_x) or np.isnan(end_y) or np.isnan(end_z)):
            ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 
                   color=segment_color, linewidth=segment_width, alpha=alpha)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Skeleton at time {time_idx}')
    
    # Set equal aspect ratio for accurate 3D representation
    ax.set_box_aspect([1, 1, 1])
    
    # Apply equal axes scaling
    set_axes_equal(ax)
    
    # Add legend
    ax.legend()
    
    return ax


def plot_skeleton_trajectory(dataset, start_time=0, end_time=None, individual_idx=0, 
                           ax=None, trajectory_alpha=0.3, current_frame_alpha=0.8,
                           keypoint_size=50, segment_width=2, arena_points=None, arena_color='lightgray'):
    """
    Plot skeleton trajectory over time with current frame highlighted.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing position data
    start_time : int, optional
        Start time index (default: 0)
    end_time : int, optional
        End time index (default: None, uses all available time)
    individual_idx : int, optional
        Individual index to plot (default: 0)
    ax : matplotlib.axes.Axes, optional
        Matplotlib 3D axes to plot on
    trajectory_alpha : float, optional
        Transparency for trajectory lines (default: 0.3)
    current_frame_alpha : float, optional
        Transparency for current frame (default: 0.8)
    keypoint_size : int, optional
        Size of keypoint markers (default: 50)
    segment_width : int, optional
        Width of skeleton segment lines (default: 2)
    arena_points : np.ndarray, optional
        Arena points to plot (shape: n_points, 3) (default: None)
    arena_color : str, optional
        Color for arena points (default: 'lightgray')
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    if end_time is None:
        end_time = dataset.sizes['time']
    
    # Plot arena points if provided
    if arena_points is not None:
        ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
                  c=arena_color, s=30, alpha=0.6, label='Arena')
    
    # Get position data for the time range
    positions = dataset.position.isel(individuals=individual_idx).sel(time=slice(start_time, end_time))
    
    # Get keypoint names and skeleton
    keypoint_names = list(dataset.coords['keypoints'].values)
    skeleton = dataset.attrs.get('skeleton', [])
    
    # Convert skeleton from keypoint names to indices
    keypoint_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    skeleton_indices = []
    for start_kp, end_kp in skeleton:
        if start_kp in keypoint_to_idx and end_kp in keypoint_to_idx:
            skeleton_indices.append((keypoint_to_idx[start_kp], keypoint_to_idx[end_kp]))
    
    # Plot trajectory for each keypoint
    for kp_idx in range(len(keypoint_names)):
        x_traj = positions.sel(keypoints=keypoint_names[kp_idx], space='x').values
        y_traj = positions.sel(keypoints=keypoint_names[kp_idx], space='y').values
        z_traj = positions.sel(keypoints=keypoint_names[kp_idx], space='z').values
        
        # Only plot valid points
        valid_mask = ~(np.isnan(x_traj) | np.isnan(y_traj) | np.isnan(z_traj))
        if np.any(valid_mask):
            ax.plot(x_traj[valid_mask], y_traj[valid_mask], z_traj[valid_mask], 
                   alpha=trajectory_alpha, linewidth=1, color='gray')
    
    # Plot current frame skeleton
    plot_skeleton_3d(dataset, time_idx=end_time-1, individual_idx=individual_idx, 
                    ax=ax, keypoint_size=keypoint_size, segment_width=segment_width,
                    keypoint_color='red', segment_color='blue', alpha=current_frame_alpha,
                    arena_points=None)  # Don't plot arena again
    
    ax.set_title(f'Skeleton trajectory (frames {start_time}-{end_time-1})')
    
    # Set equal aspect ratio for accurate 3D representation
    ax.set_box_aspect([1, 1, 1])
    
    # Apply equal axes scaling
    set_axes_equal(ax)
    
    return ax


def create_skeleton_animation(dataset, start_time=0, end_time=None, individual_idx=0,
                           interval=100, keypoint_size=50, segment_width=2, 
                           arena_points=None, arena_color='lightgray'):
    """
    Create an animated skeleton plot.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing position data
    start_time : int, optional
        Start time index (default: 0)
    end_time : int, optional
        End time index (default: None, uses all available time)
    individual_idx : int, optional
        Individual index to plot (default: 0)
    interval : int, optional
        Animation interval in milliseconds (default: 100)
    keypoint_size : int, optional
        Size of keypoint markers (default: 50)
    segment_width : int, optional
        Width of skeleton segment lines (default: 2)
    arena_points : np.ndarray, optional
        Arena points to plot (shape: n_points, 3) (default: None)
    arena_color : str, optional
        Color for arena points (default: 'lightgray')
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object
    """
    if end_time is None:
        end_time = dataset.sizes['time']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits based on data
    positions = dataset.position.isel(individuals=individual_idx).sel(time=slice(start_time, end_time))
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
    
    # Set equal aspect ratio for accurate 3D representation
    ax.set_box_aspect([1, 1, 1])
    
    def animate(frame_idx):
        ax.clear()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Plot arena points if provided
        if arena_points is not None:
            ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
                      c=arena_color, s=30, alpha=0.6, label='Arena')
        
        if np.any(valid_mask):
            ax.set_xlim(x_data[valid_mask].min(), x_data[valid_mask].max())
            ax.set_ylim(y_data[valid_mask].min(), y_data[valid_mask].max())
            ax.set_zlim(z_data[valid_mask].min(), z_data[valid_mask].max())
        
        # Set equal aspect ratio for accurate 3D representation
        ax.set_box_aspect([1, 1, 1])
        
        plot_skeleton_3d(dataset, time_idx=start_time + frame_idx, individual_idx=individual_idx,
                        ax=ax, keypoint_size=keypoint_size, segment_width=segment_width,
                        arena_points=None)  # Don't plot arena again
        ax.set_title(f'Skeleton animation - Frame {start_time + frame_idx}')
        
        # Apply equal axes scaling
        set_axes_equal(ax)
    
    anim = FuncAnimation(fig, animate, frames=end_time-start_time, 
                       interval=interval, blit=False, repeat=True)
    
    return anim


def plot_multiple_frames(dataset, time_indices, individual_idx=0, n_cols=3, 
                        figsize=(15, 12), keypoint_size=30, segment_width=1,
                        arena_points=None, arena_color='lightgray'):
    """
    Plot multiple skeleton frames in a grid layout.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing position data
    time_indices : list
        List of time indices to plot
    individual_idx : int, optional
        Individual index to plot (default: 0)
    n_cols : int, optional
        Number of columns in the grid (default: 3)
    figsize : tuple, optional
        Figure size (default: (15, 12))
    keypoint_size : int, optional
        Size of keypoint markers (default: 30)
    segment_width : int, optional
        Width of skeleton segment lines (default: 1)
    arena_points : np.ndarray, optional
        Arena points to plot (shape: n_points, 3) (default: None)
    arena_color : str, optional
        Color for arena points (default: 'lightgray')
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    n_frames = len(time_indices)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    
    for i, time_idx in enumerate(time_indices):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        plot_skeleton_3d(dataset, time_idx=time_idx, individual_idx=individual_idx,
                        ax=ax, keypoint_size=keypoint_size, segment_width=segment_width,
                        arena_points=arena_points, arena_color=arena_color)
        ax.set_title(f'Frame {time_idx}')
        # Set equal aspect ratio for accurate 3D representation
        ax.set_box_aspect([1, 1, 1])
        # Apply equal axes scaling
        set_axes_equal(ax)
    
    plt.tight_layout()
    return fig


def plot_skeleton_with_confidence(dataset, time_idx=0, individual_idx=0, ax=None,
                                confidence_threshold=0.5, keypoint_size=50, segment_width=2,
                                arena_points=None, arena_color='lightgray'):
    """
    Plot skeleton with confidence-based coloring.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing position and confidence data
    time_idx : int, optional
        Time index to plot (default: 0)
    individual_idx : int, optional
        Individual index to plot (default: 0)
    ax : matplotlib.axes.Axes, optional
        Matplotlib 3D axes to plot on
    confidence_threshold : float, optional
        Confidence threshold for coloring (default: 0.5)
    keypoint_size : int, optional
        Size of keypoint markers (default: 50)
    segment_width : int, optional
        Width of skeleton segment lines (default: 2)
    arena_points : np.ndarray, optional
        Arena points to plot (shape: n_points, 3) (default: None)
    arena_color : str, optional
        Color for arena points (default: 'lightgray')
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot arena points if provided
    if arena_points is not None:
        ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
                  c=arena_color, s=30, alpha=0.6, label='Arena')
    
    # Get position and confidence data
    positions = dataset.position.isel(time=time_idx, individuals=individual_idx)
    confidence = dataset.confidence.isel(time=time_idx, individuals=individual_idx)
    
    # Get keypoint names and skeleton
    keypoint_names = list(dataset.coords['keypoints'].values)
    skeleton = dataset.attrs.get('skeleton', [])
    
    # Convert skeleton from keypoint names to indices
    keypoint_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    skeleton_indices = []
    for start_kp, end_kp in skeleton:
        if start_kp in keypoint_to_idx and end_kp in keypoint_to_idx:
            skeleton_indices.append((keypoint_to_idx[start_kp], keypoint_to_idx[end_kp]))
    
    # Extract coordinates and confidence
    x_coords = positions.sel(space='x').values
    y_coords = positions.sel(space='y').values
    z_coords = positions.sel(space='z').values
    conf_values = confidence.values
    
    # Plot keypoints with confidence-based coloring
    valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords) | np.isnan(z_coords))
    if np.any(valid_mask):
        colors = ['red' if conf >= confidence_threshold else 'orange' 
                 for conf in conf_values[valid_mask]]
        ax.scatter(x_coords[valid_mask], y_coords[valid_mask], z_coords[valid_mask], 
                  s=keypoint_size, c=colors, alpha=0.8, label='Keypoints')
    
    # Plot skeleton segments with confidence-based coloring
    for start_idx, end_idx in skeleton_indices:
        start_x, start_y, start_z = x_coords[start_idx], y_coords[start_idx], z_coords[start_idx]
        end_x, end_y, end_z = x_coords[end_idx], y_coords[end_idx], z_coords[end_idx]
        
        # Check if both endpoints are valid
        if not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(start_z) or
                np.isnan(end_x) or np.isnan(end_y) or np.isnan(end_z)):
            # Use average confidence of endpoints for segment color
            avg_conf = (conf_values[start_idx] + conf_values[end_idx]) / 2
            segment_color = 'blue' if avg_conf >= confidence_threshold else 'orange'
            ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 
                   color=segment_color, linewidth=segment_width, alpha=0.8)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Skeleton at time {time_idx} (confidence threshold: {confidence_threshold})')
    
    # Set equal aspect ratio for accurate 3D representation
    ax.set_box_aspect([1, 1, 1])
    
    # Apply equal axes scaling
    set_axes_equal(ax)
    
    ax.legend()
    
    return ax 