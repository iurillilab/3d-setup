"""
Example script demonstrating skeleton plotting functions.

This script shows how to use the skeleton plotting utilities to visualize
3D triangulated datasets with various plotting options.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from threed_utils.io import load_triangulated_ds
from threed_utils.visualization import (
    plot_skeleton_3d,
    plot_skeleton_trajectory,
    create_skeleton_animation,
    plot_multiple_frames,
    plot_skeleton_with_confidence
)

# Load the dataset
filename = "/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021_triangulated_points_20250710-163821.h5"
ds = load_triangulated_ds(filename)

print(f"Dataset shape: {ds.position.shape}")
print(f"Keypoints: {list(ds.coords['keypoints'].values)}")
print(f"Skeleton: {ds.attrs.get('skeleton', [])}")

# Example 1: Plot single frame skeleton
print("\n1. Plotting single frame skeleton...")
fig = plt.figure(figsize=(12, 10))
ax = plot_skeleton_3d(ds, time_idx=0, keypoint_size=100, segment_width=3)
plt.show()

# Example 2: Plot skeleton trajectory
print("\n2. Plotting skeleton trajectory...")
fig = plt.figure(figsize=(12, 10))
ax = plot_skeleton_trajectory(ds, start_time=0, end_time=100, keypoint_size=80, segment_width=3)
plt.show()

# Example 3: Plot multiple frames in grid
print("\n3. Plotting multiple frames...")
time_indices = [0, 50, 100, 150, 200, 250]
fig = plot_multiple_frames(ds, time_indices, keypoint_size=50, segment_width=2)
plt.show()

# Example 4: Plot with confidence coloring (if confidence data is available)
if 'confidence' in ds.data_vars:
    print("\n4. Plotting with confidence coloring...")
    fig = plt.figure(figsize=(12, 10))
    ax = plot_skeleton_with_confidence(ds, time_idx=0, confidence_threshold=0.5, 
                                     keypoint_size=100, segment_width=3)
    plt.show()

# Example 5: Create animation (uncomment to run)
# print("\n5. Creating animation...")
# anim = create_skeleton_animation(ds, start_time=0, end_time=100, interval=200)
# plt.show()

print("\nAll examples completed!") 