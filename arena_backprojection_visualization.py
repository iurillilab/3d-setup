#!/usr/bin/env python3
"""
Script to visualize backprojected arena points on the first frame from each camera view.
Shows arena triangulation accuracy by displaying how well the 3D points project back to each 2D view.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from threed_utils.arena_utils import load_arena_multiview_ds, triangulate_arena
from threed_utils.io import read_calibration_toml
from threed_utils.multiview_calibration.geometry import project_points


def read_nth_frame(video_path: Path, n: int = 0) -> np.ndarray:
    """
    Read the nth frame from a video file.
    
    Parameters
    ----------
    video_path : Path
        Path to the video file
    n : int, optional
        Frame number to read (default: 0 for first frame)
        
    Returns
    -------
    np.ndarray
        Video frame as numpy array
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Failed to read frame {n} from {video_path}")
    
    cap.release()
    return frame


def get_video_frames(video_dir: Path, frame_number: int = 0) -> dict:
    """
    Get frames from all video files in the directory.
    
    Parameters
    ----------
    video_dir : Path
        Directory containing video files
    frame_number : int, optional
        Frame number to extract (default: 0)
        
    Returns
    -------
    dict
        Dictionary mapping camera names to video frames
    """
    video_files = list(video_dir.glob("*.mp4"))
    
    frames = {}
    for video_file in video_files:
        # Extract camera name from filename (e.g., "multicam_video_2025-05-07T14_11_04_central.mp4" -> "central")
        filename = video_file.stem
        if "_central." in video_file.name or filename.endswith("_central"):
            camera_name = "central"
        elif "_mirror-top." in video_file.name or filename.endswith("_mirror-top"):
            camera_name = "mirror-top"
        elif "_mirror-bottom." in video_file.name or filename.endswith("_mirror-bottom"):
            camera_name = "mirror-bottom"
        elif "_mirror-left." in video_file.name or filename.endswith("_mirror-left"):
            camera_name = "mirror-left"
        elif "_mirror-right." in video_file.name or filename.endswith("_mirror-right"):
            camera_name = "mirror-right"
        else:
            print(f"Warning: Could not identify camera for {video_file.name}")
            continue
            
        frame = read_nth_frame(video_file, frame_number)
        frames[camera_name] = frame
        
    return frames


def backproject_arena_points(arena_3d_ds, calib_toml_path: Path) -> dict:
    """
    Backproject 3D arena points to all camera views.
    
    Parameters
    ----------
    arena_3d_ds : xr.Dataset
        Dataset containing triangulated 3D arena points
    calib_toml_path : Path
        Path to calibration TOML file
        
    Returns
    -------
    dict
        Dictionary mapping camera names to 2D projected points
    """
    # Load calibration data
    cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)
    
    # Extract 3D arena points (shape: 8 corners, 3 coordinates)
    arena_points_3d = arena_3d_ds.position.isel(time=0, individuals=0).values.T  # Shape: (8, 3)
    
    # Backproject to each camera
    backprojected_points = {}
    
    for i, cam_name in enumerate(cam_names):
        # Get calibration parameters for this camera
        extr = extrinsics[i]  # (6,) rotation vector + translation
        cam_matrix, dist_coefs = intrinsics[i]  # (3,3) camera matrix and distortion coefficients
        
        # Project 3D points to 2D
        points_2d = project_points(arena_points_3d, extr, cam_matrix, dist_coefs)
        backprojected_points[cam_name] = points_2d
        
    return backprojected_points


def plot_arena_backprojections(frames: dict, backprojected_points: dict, 
                              original_2d_points: dict = None, 
                              figsize: tuple = (20, 12)):
    """
    Plot video frames with backprojected arena points overlaid.
    
    Parameters
    ----------
    frames : dict
        Dictionary mapping camera names to video frames
    backprojected_points : dict
        Dictionary mapping camera names to backprojected 2D points
    original_2d_points : dict, optional
        Dictionary mapping camera names to original 2D arena coordinates for comparison
    figsize : tuple, optional
        Figure size (default: (20, 12))
    """
    n_cameras = len(frames)
    n_cols = 3
    n_rows = (n_cameras + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    camera_names = sorted(frames.keys())
    
    for i, camera_name in enumerate(camera_names):
        ax = axes[i]
        
        # Display the frame
        frame = frames[camera_name]
        if len(frame.shape) == 3:  # Color image
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:  # Grayscale
            ax.imshow(frame, cmap='gray')
        
        # Plot backprojected arena points
        if camera_name in backprojected_points:
            points_2d = backprojected_points[camera_name]
            ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                      c='red', s=30, marker='o', alpha=0.8, 
                      label='Backprojected 3D')
            
            # Number the corners
            for j, (x, y) in enumerate(points_2d):
                ax.annotate(f'{j}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', color='red', 
                           fontsize=8, fontweight='bold')
        
        # Plot original 2D points for comparison if provided
        if original_2d_points and camera_name in original_2d_points:
            orig_points = np.array(original_2d_points[camera_name])
            ax.scatter(orig_points[:, 0], orig_points[:, 1], 
                      c='blue', s=30, marker='x', alpha=0.8,
                      label='Original 2D')
        
        ax.set_title(f'{camera_name}')
        ax.axis('off')
        if i == 0:  # Add legend to first subplot
            ax.legend(loc='upper right')
    
    # Hide unused subplots
    for i in range(len(camera_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Arena Backprojections on First Frame', fontsize=16, y=0.98)
    
    # Save the plot
    plt.savefig('arena_backprojections.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'arena_backprojections.png'")
    
    plt.show()
    
    return fig


def main():
    """Main function to run the arena backprojection visualization."""
    
    # Paths (update these as needed)
    arena_json_path = Path("/Users/vigji/Desktop/test_3d/multicam_video_2025-05-07T10_12_11_20250528-153946.json")
    calib_toml_path = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328/mc_calibration_output_20250710-152443/calibration_from_mc.toml")
    video_dir = Path("/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021")
    
    print("Loading arena coordinates...")
    arena_multiview_ds = load_arena_multiview_ds(arena_json_path)
    
    print("Triangulating arena points...")
    arena_3d_ds = triangulate_arena(arena_multiview_ds, calib_toml_path)
    
    print("Loading video frames...")
    frames = get_video_frames(video_dir, frame_number=0)
    print(f"Loaded {len(frames)} camera views: {list(frames.keys())}")
    
    print("Backprojecting 3D arena points...")
    backprojected_points = backproject_arena_points(arena_3d_ds, calib_toml_path)
    
    # Also get original 2D points for comparison
    original_2d_points = {}
    arena_coords = arena_multiview_ds.position.isel(time=0, individuals=0).values
    for i, view_name in enumerate(arena_multiview_ds.view.values):
        original_2d_points[view_name] = arena_coords[i, :, :].T  # Shape: (8, 2)
    
    print("Creating visualization...")
    fig = plot_arena_backprojections(frames, backprojected_points, original_2d_points)
    
    print("Calculating reprojection errors...")
    total_error = 0
    n_points = 0
    for camera_name in backprojected_points:
        if camera_name in original_2d_points:
            backproj = backprojected_points[camera_name]
            original = original_2d_points[camera_name]
            errors = np.linalg.norm(backproj - original, axis=1)
            mean_error = np.mean(errors)
            print(f"{camera_name}: Mean reprojection error = {mean_error:.2f} pixels")
            total_error += np.sum(errors)
            n_points += len(errors)
    
    overall_mean_error = total_error / n_points if n_points > 0 else 0
    print(f"\nOverall mean reprojection error: {overall_mean_error:.2f} pixels")
    
    return fig


if __name__ == "__main__":
    main() 