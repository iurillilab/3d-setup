# %%
%matplotlib widget
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import multicam_calibration.geometry as mcc_geom
import numpy as np
import pandas as pd
import xarray as xr
from movement.io.load_poses import from_numpy
from threed_utils.io import movement_ds_from_anipose_triangulation_df, read_calibration_toml

from threed_utils.anipose.triangulate import CameraGroup, triangulate_core


data_dir = Path("/Users/vigji/Desktop/test-anipose/cropped_calibration_vid")

# Read checkerboard detections as movement dataset

# Load last available calibration among mc_calibrarion_output_*
calibration_paths = sorted(data_dir.glob("mc_calibration_output_*"))
last_calibration_path = calibration_paths[-1]

all_calib_uvs = np.load(last_calibration_path / "all_calib_uvs.npy")
calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)

print(all_calib_uvs.shape)

# Temporary patch for loading checkerboard data as dataset:
views_dss = []
for n_view, view in enumerate(cam_names):
    # position_array = np.moveaxis(all_calib_uvs[n_view], 0, -1)  # Move views to last axis
    position_array = all_calib_uvs[n_view]
    position_array = position_array[:, np.newaxis, :, :]  # Add individuals axis
    confidence_array = np.ones(position_array.shape[:-1])

    keypoint_names = [str(i) for i in range(position_array.shape[2])]
    individual_names = ["checkerboard"]
    source_software = "opencv"

    views_dss.append(from_numpy(position_array=position_array, 
                                confidence_array=confidence_array,
                                individual_names=individual_names,
                                keypoint_names=keypoint_names,
                                source_software=source_software,
                                ))
    
new_coord_views = xr.DataArray(cam_names, dims="view")

views_ds = xr.concat(views_dss, dim=new_coord_views)

time_slice = slice(145, 1000)
views_ds = views_ds.sel(time=time_slice, drop=True)


# %%
def mcc_triangulate_ds(
    xarray_dataset, calib_toml_path, progress_bar=True
):
    cam_names, _, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)

    positions = xarray_dataset.position
    confidence = xarray_dataset.confidence  # TODO implement confidence propagation

    # use cam_names to sort the view axis, after having checked that the views are the same:
    print(xarray_dataset.coords["view"].values, cam_names)
    assert set(xarray_dataset.coords["view"].values) == set(cam_names), "Views do not match: " + str(list(positions.coords["view"])) + " vs " + str(cam_names)
    positions = positions.sel(view=cam_names)
    print(positions.coords["view"].values)
    
    # get first individual, regarless of its name:
    positions = positions.sel(individuals=positions.coords["individuals"][0], drop=True)

    # enforce order:
    positions = positions.transpose("view", "time", "keypoints", "space").values
    all_triang = []
    for i in tqdm(range(len(xarray_dataset.coords["keypoints"])), "Triangulating keypoints: ", 
                  disable=not progress_bar):
        triang = mcc_geom.triangulate(positions[:, :, i, :], extrinsics, intrinsics)
        all_triang.append(triang)

    threed_coords = np.array(all_triang)  # shape n_keypoints, n_frames, 3
    # reshape to n_frames, 1, n_keypoints, 3
    threed_coords = threed_coords.transpose(1, 0, 2)[:, np.newaxis, :, :]
    # TODO propagate confidence smartly
    confidence_array = np.ones(threed_coords.shape[:-1])

    return from_numpy(position_array=threed_coords,
               confidence_array=confidence_array,
               individual_names=xarray_dataset.coords["individuals"].values,
               keypoint_names=xarray_dataset.coords["keypoints"].values,
               source_software=xarray_dataset.attrs["source_software"] + "_triangulated",
               )

mcc_triangulated_ds = mcc_triangulate_ds(views_ds, calib_toml_path)

# %%
# ===============================================
# Test anipose triangulation
# ===============================================

def anipose_triangulate_ds(views_ds, calib_toml_path, **config_kwargs):
    triang_config = config_kwargs
    config = dict(triangulation=triang_config)

    calib_fname = str(calib_toml_path)
    cgroup = CameraGroup.load(calib_fname)

    individual_name = views_ds.coords["individuals"][0]
    reshaped_ds = views_ds.sel(individuals=individual_name, time=time_slice).transpose("view", "time", "keypoints", "space")
    positions = reshaped_ds.position.values
    scores = reshaped_ds.confidence.values

    triang_df = triangulate_core(config, 
                 positions, 
                 scores, 
                 views_ds.coords["keypoints"].values, 
                 cgroup, 
                 )
    return movement_ds_from_anipose_triangulation_df(triang_df)


triang_config = {
    "ransac": True,
    "optim": False,
}
anipose_triangulated_ds = anipose_triangulate_ds(views_ds, calib_toml_path, **triang_config)

triang_config_optim = {
    "ransac": True,
    "optim": True,
    "optim_chunking": True,
    "optim_chunking_size": 100,
    "score_threshold": 0.0,
    "scale_smooth": 4,
    "scale_length": 2,
    "scale_length_weak": 0.5,
    "n_deriv_smooth": 2,
    "reproj_error_threshold": 150,
    "constraints": [], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    "constraints_weak": [], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
}
# de_nanned = views_ds.copy()
# de_nanned.position.values[np.isnan(de_nanned.position.values)] = 0

# subtract random val between 0 and 1 to confidences:
# de_nanned.confidence.values -= np.random.rand(*de_nanned.confidence.values.shape)
anipose_triangulated_ds_optim = anipose_triangulate_ds(views_ds, 
                                                       calib_toml_path, 
                                                       **triang_config_optim)

# %%
def plot_3d_points_and_trail(coords_array, ax=None, individual_name="checkerboard", 
                             trail=True, frame_idx=None):
    """Plot 3D points for a single frame and the trail of first point over time"""
    coords_array = coords_array.position.sel(individuals=individual_name).values

    if frame_idx is None:
        non_nan_idxs = np.where(~np.isnan(coords_array).any(axis=(1, 2)))[0]
        frame_idx = non_nan_idxs[0]

    frame_points = coords_array[frame_idx, :, :]
    point_trail = coords_array[:, 0, :]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure

    ax.scatter(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2])
    if trail:
        ax.scatter(point_trail[:, 0], point_trail[:, 1], point_trail[:, 2], s=5)

    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.axis("equal")

    return fig

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for index in [0]:
    trail = True
    plot_3d_points_and_trail(mcc_triangulated_ds, ax=ax, frame_idx=index, trail=trail)
    plot_3d_points_and_trail(anipose_triangulated_ds, ax=ax, frame_idx=index, trail=trail)
    plot_3d_points_and_trail(anipose_triangulated_ds_optim, ax=ax, frame_idx=index, trail=trail)
    plt.show()


# %%
# ===============================================
# Sleap coordinates triangulation
# ===============================================
# data_dir = Path("/Users/vigji/Desktop/test-anipose")
import re
from movement.io.load_poses import from_file
print(data_dir)
slp_files_dir = data_dir.parent / "test_slp_files"
slp_files = list(slp_files_dir.glob("*.slp"))
print(slp_files)
cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([\w-]+)\.\w+(?:\.\w+)+$"
file_path_dict = {re.search(cam_regex, str(f.name)).groups()[0]: f for f in slp_files}
# From movement.io.load_poses.from_multiview_files, split out here just to fix uppercase inconsistency bug:
views_list = list(file_path_dict.keys())
new_coord_views = xr.DataArray(views_list, dims="view")

dataset_list = [
    from_file(f, source_software="SLEAP")
    for f in file_path_dict.values()
]
# make coordinates labels of the keypoints axis all lowercase
for ds in dataset_list:
    ds.coords["keypoints"] = ds.coords["keypoints"].str.lower()

time_slice = slice(0, 1000)
ds = xr.concat(dataset_list, dim=new_coord_views).sel(time=time_slice)

bodyparts = list(ds.coords["keypoints"].values)

print(ds.position.shape, ds.confidence.shape, bodyparts)

triang_config_optim = {
    "ransac": True,
    "optim": True,
    "optim_chunking": True,
    "optim_chunking_size": 100,
    "score_threshold": 0.0,
    "scale_smooth": 4,
    "scale_length": 2,
    "scale_length_weak": 0.5,
    "n_deriv_smooth": 2,
    "reproj_error_threshold": 150,
    "constraints": [], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    "constraints_weak": [], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
}

mcc_triangulated_ds = mcc_triangulate_ds(ds, calib_toml_path)

anipose_triangulated_ds_optim = anipose_triangulate_ds(ds, 
                                                       calib_toml_path, 
                                                       **triang_config_optim)



# %%
ds
# %%