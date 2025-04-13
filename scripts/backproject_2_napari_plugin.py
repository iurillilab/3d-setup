
from movement.io.save_poses import to_dlc_file
from movement.io.load_poses import from_numpy
from pathlib import Path
import xarray as xr
import numpy as np

file = Path("/Users/vigji/Downloads/multicam_video_2024-08-07T15_48_36_cropped_20250325101012/reprojection_multicam_video_2024-08-07T15_48_36_cropped_20250325101012_triangulated_points_20250330-234823.h5")

# read file from netcdf
ds = xr.open_dataset(file)

CLIP_AT = 2000

for camera in ds.coords["camera"].values:
    sub_ds = ds.sel(camera=camera, drop=True)

    sub_ds.reprojections.values = np.clip(sub_ds.reprojections.values, 
                                          -CLIP_AT, 
                                          CLIP_AT)

    mov_ds = from_numpy(position_array=sub_ds.reprojections.transpose("time", "xy", "keypoints").values[:, :, :, None],
               confidence_array=sub_ds.confidence.transpose("time", "keypoints").values[:, :, None],
               keypoint_names=sub_ds.keypoints.values,
    )
    new_filename = file.stem.split("reprojection_")[1].split("cropped")[0] + f"_{camera}_reprojection.h5"
    
    to_dlc_file(mov_ds, file.parent / new_filename, split_individuals=False,
                )
