# %%
%matplotlib widget
from utils import read_calibration_toml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import multicam_calibration.geometry as mcc_geom
import numpy as np
import xarray as xr
from movement.io.load_poses import from_numpy


data_dir = Path("/Users/vigji/Desktop/test-anipose/cropped_calibration_vid")

# Load last available calibration among mc_calibrarion_output_*
calibration_paths = sorted(data_dir.glob("mc_calibration_output_*"))
last_calibration_path = calibration_paths[-1]

all_calib_uvs = np.load(last_calibration_path / "all_calib_uvs.npy")
calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)

print(all_calib_uvs.shape)

# %%
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

views_ds


# %%
def triangulate_all_keypoints(
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

# %%
checkboard_3d = triangulate_all_keypoints(views_ds, calib_toml_path)

# %%
def plot_3d_points_and_trail(coords_array, individual_name="checkerboard", trail=True):
    """Plot 3D points for a single frame and the trail of first point over time"""
    coords_array = coords_array.position.sel(individuals=individual_name).values
    non_nan_idxs = np.where(~np.isnan(coords_array).any(axis=(1, 2)))[0]
    frame_idx = non_nan_idxs[0]
    frame_points = coords_array[frame_idx, :, :]
    point_trail = coords_array[:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2])
    if trail:
        ax.scatter(point_trail[:, 0], point_trail[:, 1], point_trail[:, 2], s=5)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.axis("equal")
    return fig

# %%
fig = plot_3d_points_and_trail(checkboard_3d)
plt.show()

# %%
other_toml_to_test = "/Users/vigji/Desktop/test-anipose/cropped_calibration_vid/anipose_calib/calibration.toml"
checkboard_anipose = triangulate_all_keypoints(views_ds, other_toml_to_test)

# %%
fig = plot_3d_points_and_trail(checkboard_anipose, trail=False)
plt.show()

# %%

# ===============================================
# Test anipose triangulation
# ===============================================
from threed_utils.anipose.triangulate import CameraGroup, triangulate_core
calib_config = dict(board_type="checkerboard",
                board_size=(5, 7),
                board_square_side_length=12.5,
                animal_calibration=False,
                calibration_init=None,
                fisheye=False)

triang_config = {
    # "cam_regex": r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([\w-]+)\.\w+(?:\.\w+)+$",
    "ransac": False,
    "optim": False,
    "optim_chunking": True,
    "optim_chunking_size": 100,
    "score_threshold": 0.0,
}
manual_verification_config = dict(manually_verify=False)

config = dict(calibration=calib_config, 
              triangulation=triang_config,
              manual_verification=manual_verification_config,
              )

calib_fname = str(calib_toml_path)
cgroup = CameraGroup.load(calib_fname)

output_fname = data_dir / "test_triangulation_anypose.csv"
time_slice = slice(0, 1000)
reshaped_ds = views_ds.sel(individuals="checkerboard", time=time_slice).transpose("view", "time", "keypoints", "space")
positions = reshaped_ds.position.values
scores = reshaped_ds.confidence.values
# (n_cams, n_frames, n_joints, 2)
triang_df = triangulate_core(config, 
                 positions, 
                 scores, 
                 views_ds.coords["keypoints"].values, 
                 cgroup, 
                 output_fname)


# %%
triang_df.columns
# %%
# Reshape dataframe with columns keypoint1_x, keypoint1_y, keypoint1_z, keypoint1_confidence_score, 
# keypoint2_x, keypoint2_y, keypoint2_z, keypoint2_confidence_score, ...
# to array of positions with dimensions time, individuals, keypoints, space,
# and array of confidence scores with dimensions time, individuals, keypoints

# Get list of unique keypoint names by removing _x, _y, _z, _error, _ncams, _score suffixes

keypoint_names = sorted(list(set([col.rsplit('_', 1)[0] for col in triang_df.columns 
                               if any(col.endswith(f'_{s}') for s in ['x','y','z'])])))

n_frames = len(triang_df)
n_keypoints = len(keypoint_names)

# Initialize arrays and fill
position_array = np.zeros((n_frames, 1, n_keypoints, 3))  # 1 for single individual
confidence_array = np.zeros((n_frames, 1, n_keypoints))
for i, kp in enumerate(keypoint_names):
    for j, coord in enumerate(['x', 'y', 'z']):
        position_array[:, 0, i, j] = triang_df[f'{kp}_{coord}']
    confidence_array[:, 0, i] = triang_df[f'{kp}_score']

individual_names = ['checkerboard']

anipose_triang_ds = from_numpy(position_array=position_array, 
                                confidence_array=confidence_array,
                                individual_names=individual_names,
                                keypoint_names=keypoint_names,
                                source_software="'anipose_triangulation",
                                )

# triang_ds
# %%
fig = plot_3d_points_and_trail(anipose_triang_ds, trail=True)
plt.show()

# %%
# %%



# from threed_utils.anipose.common import get_cam_name
from threed_utils.anipose.triangulate import triangulate_core, CameraGroup
from movement.io.load_poses import from_file, from_multiview_files
import pickle
import re
import numpy as np
import pandas as pd


from pathlib import Path

data_dir = Path("/Users/vigji/Desktop/test-anipose")
calibration_dir = data_dir / "cropped_calibration_vid" / "anipose_calib"
slp_files_dir = data_dir / "test_slp_files"
slp_files = list(slp_files_dir.glob("*.slp"))
file_path_dict = {re.search(triang_config['cam_regex'], str(f.name)).groups()[0]: f for f in slp_files}
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
ds = xr.concat(dataset_list, dim=new_coord_views).sel(time=time_slice, individuals="individual_0", drop=True)

bodyparts = list(ds.coords["keypoints"].values)

print(ds.position.shape, ds.confidence.shape, bodyparts)


# fig.savefig(output_dir / "triangulated_frame.png")


# # %matplotlib widget
# import matplotlib
# matplotlib.use('Agg')  # Configure backend before importing pyplot
# import matplotlib.pyplot as plt
# import numpy as np

# import multicam_calibration as mcc
# from pathlib import Path
# from tqdm import tqdm
# import pickle
# from datetime import datetime
# import flammkuchen as fl
# import toml

# import tqdm
# import cv2


# file = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\calibration_retest\Calibration\multicam_video_2024-08-03T16_36_58_20241111-164517.json"

# import json

# with open(file) as f:
#     d = json.load(f)
#     print(d)
# coordinates = d[-1]
# coordinates_ars = {}
# for key, value in coordinates["points_coordinate"].items():
#     coordinates_ars[key] = np.array(value)

# right_arr = [
#     coordinates_ars[cam] for cam in camera_sequence
# ]  # [coordinates_ars['central'], coordinates_ars['mirror-bottom'], coordinates_ars['mirror-left'], coordinates_ars['mirror-right'], coordinates_ars['mirror-top']]
# right_arr = np.stack([np.expand_dims(np.array(arr), 0) for arr in right_arr], 0)

# arena_3d = triangulate_all_keypoints(
#     right_arr[:, :, :, [1, 0]], adj_extrinsics, adj_intrinsics
# )
# # arena_triangulation = []
# # for i in tqdm(range(8)):
# #     arena_triangulation.append(mcc.triangulate(right_arr2[:, :, i, [1, 0]], adj_extrinsics, adj_intrinsics))
# # arena_triangulation = np.array(arena_triangulation).squeeze()
# arena_frame = arena_3d[:, 0, :]

# %matplotlib widget
# import matplotlib.pyplot as plt
# import numpy as np


# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")


# ax.scatter(
#     checkboard_frame[:, 0], checkboard_frame[:, 1], checkboard_frame[:, 2], c="b"
# )
# ax.scatter(arena_frame[:, 0], arena_frame[:, 1], arena_frame[:, 2], c="r")
# ax.set_xlabel("X Label")
# ax.set_ylabel("Y Label")
# ax.set_zlabel("Z Label")
# plt.axis("equal")
# plt.show()

# back_projections = {}
# names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
# for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
#     back_projections[names.pop(0)] = mcc.project_points(
#         arena_frame, extrinsic, intrinsic[0], dist_coefs=intrinsic[1]
#     )

# names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
# checkboard_back_projections = {}

# for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
#     checkboard_back_projections[names.pop(0)] = mcc.project_points(
#         checkboard_frame, extrinsic, intrinsic[0], dist_coefs=intrinsic[1]
#     )

# # get the same frame from all the videos:
# import cv2

# idx = frame_idx
# frames = {}
# names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
# for frame, name in zip(video_paths, names):
#     cap = cv2.VideoCapture(str(frame))
#     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#     res, frame = cap.read()
#     frames[name] = frame

# plt.figure(figsize=(10, 10))

# for i, view in enumerate(camera_sequence):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(frames[view], cmap="gray")
#     plt.scatter(back_projections[view][:, 0], back_projections[view][:, 1], c="r", s=10)
#     plt.scatter(
#         checkboard_back_projections[view][:, 0],
#         checkboard_back_projections[view][:, 1],
#         c="b",
#         s=10,
#     )
#     plt.title(view)

# plt.show()

# back_projections_nodist = {}
# names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
# for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
#     back_projections_nodist[names.pop(0)] = mcc.project_points(
#         arena_triangulation, extrinsic, intrinsic[0], dist_coefs=None
#     )

# names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
# checkboard_back_projections_nodist = {}

# for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
#     checkboard_back_projections_nodist[names.pop(0)] = mcc.project_points(
#         checkboard_3d, extrinsic, intrinsic[0], dist_coefs=None
#     )

# # save arena_points, tracked_points for testing
# data = {
#     "arena_coordinates": right_arr,
#     "tracked_points": tracked_points[:, :200, :, :],
#     "checkboard_points": all_calib_uvs[:, :200, :, :],
# }
# calib_data = {
#     "points": data,
#     "extrinsics": adj_extrinsics,
#     "intrinsics": adj_intrinsics,
# }
# pickle.dump(calib_data, open(r"../tests/assets/arena_tracked_points.pkl", "wb"))
# %%
