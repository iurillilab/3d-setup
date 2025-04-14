from movement.io.save_poses import to_dlc_file
from movement.io.load_poses import from_numpy
from pathlib import Path
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

filt = True  # if True we are looking at filtered reprojections, expecting the filename to start with "filtreprojection" 
# (easily changed/ignored if needed)
file_str = "/Users/vigji/Downloads/multicam_video_2024-08-07T15_48_36/reprojection_multicam_video_2024-08-07T15_48_36_cropped_20250325101012_triangulated_points_20250330-234823.h5"

if filt:
    file_str = file_str.replace("reprojection", "filtreprojection")

file = Path(file_str)


# read file from netcdf
ds = xr.open_dataset(file)

CLIP_AT = 2000

for camera in ds.coords["camera"].values:
    
    sub_ds = ds.sel(camera=camera, drop=True)
    vals = sub_ds.reprojections.values
    vals = np.clip(vals, -CLIP_AT, CLIP_AT)
    vals[np.isnan(vals)] = -3000

    sub_ds.reprojections.values = vals

    mov_ds = from_numpy(position_array=sub_ds.reprojections.transpose("time", "xy", "keypoints").values[:, :, :, None],
                        confidence_array=sub_ds.confidence.transpose("time", "keypoints").values[:, :, None],
                        keypoint_names=sub_ds.keypoints.values,
    )
    suffix = f"{camera}_reprojection.h5" if not filt else f"{camera}_reprojection_filtered.h5"
    new_filename = file.stem.split("reprojection_")[1].split("cropped")[0] + suffix
    # check if therre are nan values in the mov_ds
    if np.isnan(mov_ds.position).any():
        print(f"NaN values in {camera}")
        continue

    sel_k = list(mov_ds.coords["keypoints"].values).copy()

    for k in ["lblimb", "rblimb", "lflimb", "rflimb", "blimbmid", "flimbmid"]:
        sel_k.pop(sel_k.index(k))

    mov_ds = mov_ds.sel(keypoints=sel_k, drop=True)
    
    full_path = file.parent / new_filename
    print(full_path)
    if full_path.exists():
        full_path.unlink()

    to_dlc_file(mov_ds, full_path, split_individuals=False,
                )