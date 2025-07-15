#!/usr/bin/env python3
"""
Test script for arena triangulation and visualization.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from threed_utils.arena_utils import load_arena_coordinates, triangulate_arena, get_arena_points_from_dataset
from threed_utils.io import load_calibration
from threed_utils.visualization.skeleton_plots import set_axes_equal


def test_arena_triangulation():
    """Test arena triangulation and visualization."""
    
    # Load calibration data
    calibration_dir = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328/")
    cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calibration_dir)
    
    # Load arena coordinates
    arena_json_path = Path("/Users/vigji/Desktop/test_3d/multicam_video_2025-05-07T10_12_11_20250528-153946.json")
    arena_coordinates = load_arena_coordinates(arena_json_path)
    
    print("Arena coordinates loaded:")
    for cam_name, coords in arena_coordinates.items():
        print(f"  {cam_name}: {len(coords)} points")
    
    # Triangulate arena
    print("\nTriangulating arena...")
    arena_ds = triangulate_arena(arena_coordinates, calib_toml_path, cam_names)
    
    print(f"Arena dataset shape: {arena_ds.position.shape}")
    print(f"Arena keypoints: {list(arena_ds.coords['keypoints'].values)}")
    
    # Get arena points for visualization
    arena_points = get_arena_points_from_dataset(arena_ds, 0)
    print(f"Arena points shape: {arena_points.shape}")
    
    # Visualize arena
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot arena points
    ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
              c='blue', s=50, alpha=0.8, label='Arena')
    
    # Add point labels
    for i, point in enumerate(arena_points):
        ax.text(point[0], point[1], point[2], f'{i}', fontsize=8)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Triangulated Arena Points')
    
    # Apply equal axes scaling
    set_axes_equal(ax)
    
    plt.legend()
    plt.show()
    
    print("Arena triangulation test completed successfully!")
    
    return arena_ds


if __name__ == "__main__":
    test_arena_triangulation() 