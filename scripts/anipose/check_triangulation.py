# %%
# %matplotlib widget
from matplotlib import pyplot as plt
from threed_utils.io import load_triangulated_ds
from pathlib import Path
import xarray
import numpy as np
from movement.filtering import filter_by_confidence
from threed_utils.visualization.skeleton_plots import create_skeleton_animation


filename = "/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021_triangulated_points_20250710-163821.h5"
filename = Path(filename)
ds = load_triangulated_ds(filename)
ds_filt = ds.copy()
ds_filt['position'] = filter_by_confidence(ds.position, ds.confidence, threshold=0.1)

# %%
plt.figure()
plt.plot(ds.confidence.squeeze())
# %%
plt.figure()
ax = plt.axes([0, 0, 1, 1], projection="3d")
ax.scatter(ds_filt.position.sel(space="x"), ds_filt.position.sel(space="y"), ds_filt.position.sel(space="z"))
plt.show()
# %%
animation = create_skeleton_animation(ds_filt, start_time=0, end_time=300, interval=1)

animation_filename = filename.with_suffix(".mp4")
animation.save(animation_filename, writer="ffmpeg", fps=10)
# %%