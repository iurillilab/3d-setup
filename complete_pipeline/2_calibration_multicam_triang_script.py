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
calibration_path = calibration_paths[-1]
cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calibration_path / "calibration_from_mc.toml")
all_calib_uvs = np.load(calibration_path / "all_calib_uvs.npy")
print(all_calib_uvs.shape)
print(cam_names, img_sizes, extrinsics, intrinsics)

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
    calib_uvs, adj_extrinsics, adj_intrinsics, progress_bar=True
):
    all_triang = []
    progbar = tqdm if progress_bar else lambda x: x
    for i in progbar(range(calib_uvs.shape[2])):
        all_triang.append(
            mcc_geom.triangulate(calib_uvs[:, :, i, :], adj_extrinsics, adj_intrinsics)
        )

    return np.array(all_triang)

checkboard_3d = triangulate_all_keypoints(all_calib_uvs, extrinsics, intrinsics)

non_nan_idxs = np.where(~np.isnan(checkboard_3d).any(axis=(0, 2)))[0]
frame_idx = non_nan_idxs[0]
checkboard_frame = checkboard_3d[:, frame_idx, :]
checkboard_trail = checkboard_3d[0, :, :]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(checkboard_frame[:, 0], checkboard_frame[:, 1], checkboard_frame[:, 2])
ax.scatter(checkboard_trail[:, 0], checkboard_trail[:, 1], checkboard_trail[:, 2], s=5)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
plt.axis("equal")
plt.show()
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