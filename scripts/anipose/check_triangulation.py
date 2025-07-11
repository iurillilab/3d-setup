# %%
# %matplotlib widget
from matplotlib import pyplot as plt
from threed_utils.io import load_triangulated_ds, load_calibration, create_2d_ds
from pathlib import Path
import xarray
import numpy as np
from movement.filtering import filter_by_confidence

from threed_utils.visualization.skeleton_plots import create_skeleton_animation, plot_skeleton_3d, set_axes_equal
from threed_utils.arena_utils import get_triangulated_arena_ds, get_arena_points_from_dataset

filename="/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021_triangulated_points_20250711-173845.h5"
filename = Path(filename)
data_path = filename.parent
expected_views = ("central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top")


source_ds = create_2d_ds(data_path, expected_views, software="DeepLabCut", max_n_frames=1000)
print(source_ds)
print(source_ds.coords)
print(source_ds.position.shape)

# %%
assert filename.exists()
ds = load_triangulated_ds(filename)
ds_filt = ds.copy()
print(ds)
print(ds.coords)
print(ds.position.shape)
ds_filt['position'] = filter_by_confidence(ds.position, ds.confidence, threshold=0.1)

# Load calibration data
calibration_dir = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328/")
cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calibration_dir)

# Load and triangulate arena
arena_json_path = Path("/Users/vigji/Desktop/test_3d/multicam_video_2025-05-07T10_12_11_20250528-153946.json")
arena_ds = get_triangulated_arena_ds(arena_json_path, calib_toml_path)

# %%
plt.figure()
plt.plot(ds.confidence.squeeze())

# %%
# Plot mouse data with proper 3D axes scaling
plt.figure()
ax = plt.axes([0, 0, 1, 1], projection="3d")
ax.scatter(ds_filt.position.sel(space="x"), ds_filt.position.sel(space="y"), ds_filt.position.sel(space="z"))
set_axes_equal(ax)
plt.show()

# %%
# Plot arena and mouse together
arena_points = get_arena_points_from_dataset(arena_ds, 0)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot arena points
ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
          c='lightgray', s=50, alpha=0.8, label='Arena')

# Plot mouse skeleton
plot_skeleton_3d(ds_filt, time_idx=0, individual_idx=0, ax=ax, arena_points=None)
ax.set_title('Arena and Mouse - Frame 0')
set_axes_equal(ax)
plt.show()

# %%
# Create animation with arena
animation = create_skeleton_animation(ds_filt, start_time=0, end_time=300, interval=1,
                                   arena_points=arena_points, arena_color='lightgray')

animation_filename = filename.with_suffix(".mp4")
animation.save(animation_filename, writer="ffmpeg", fps=60)
# %%