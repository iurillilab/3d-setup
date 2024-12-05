# %matplotlib widget
import matplotlib
matplotlib.use('Agg')  # Configure backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

import multicam_calibration as mcc
from pathlib import Path
from tqdm import tqdm
import pickle
from datetime import datetime
import flammkuchen as fl
import toml

import tqdm
import cv2


file = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\calibration_retest\Calibration\multicam_video_2024-08-03T16_36_58_20241111-164517.json"

import json

with open(file) as f:
    d = json.load(f)
    print(d)
coordinates = d[-1]
coordinates_ars = {}
for key, value in coordinates["points_coordinate"].items():
    coordinates_ars[key] = np.array(value)

right_arr = [
    coordinates_ars[cam] for cam in camera_sequence
]  # [coordinates_ars['central'], coordinates_ars['mirror-bottom'], coordinates_ars['mirror-left'], coordinates_ars['mirror-right'], coordinates_ars['mirror-top']]
right_arr = np.stack([np.expand_dims(np.array(arr), 0) for arr in right_arr], 0)

arena_3d = triangulate_all_keypoints(
    right_arr[:, :, :, [1, 0]], adj_extrinsics, adj_intrinsics
)
# arena_triangulation = []
# for i in tqdm(range(8)):
#     arena_triangulation.append(mcc.triangulate(right_arr2[:, :, i, [1, 0]], adj_extrinsics, adj_intrinsics))
# arena_triangulation = np.array(arena_triangulation).squeeze()
arena_frame = arena_3d[:, 0, :]

%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection="3d")


ax.scatter(
    checkboard_frame[:, 0], checkboard_frame[:, 1], checkboard_frame[:, 2], c="b"
)
ax.scatter(arena_frame[:, 0], arena_frame[:, 1], arena_frame[:, 2], c="r")
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
plt.axis("equal")
plt.show()

back_projections = {}
names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
    back_projections[names.pop(0)] = mcc.project_points(
        arena_frame, extrinsic, intrinsic[0], dist_coefs=intrinsic[1]
    )

names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
checkboard_back_projections = {}

for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
    checkboard_back_projections[names.pop(0)] = mcc.project_points(
        checkboard_frame, extrinsic, intrinsic[0], dist_coefs=intrinsic[1]
    )

# get the same frame from all the videos:
import cv2

idx = frame_idx
frames = {}
names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
for frame, name in zip(video_paths, names):
    cap = cv2.VideoCapture(str(frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    res, frame = cap.read()
    frames[name] = frame

plt.figure(figsize=(10, 10))

for i, view in enumerate(camera_sequence):
    plt.subplot(2, 3, i + 1)
    plt.imshow(frames[view], cmap="gray")
    plt.scatter(back_projections[view][:, 0], back_projections[view][:, 1], c="r", s=10)
    plt.scatter(
        checkboard_back_projections[view][:, 0],
        checkboard_back_projections[view][:, 1],
        c="b",
        s=10,
    )
    plt.title(view)

plt.show()

back_projections_nodist = {}
names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
    back_projections_nodist[names.pop(0)] = mcc.project_points(
        arena_triangulation, extrinsic, intrinsic[0], dist_coefs=None
    )

names = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
checkboard_back_projections_nodist = {}

for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
    checkboard_back_projections_nodist[names.pop(0)] = mcc.project_points(
        checkboard_3d, extrinsic, intrinsic[0], dist_coefs=None
    )

# save arena_points, tracked_points for testing
data = {
    "arena_coordinates": right_arr,
    "tracked_points": tracked_points[:, :200, :, :],
    "checkboard_points": all_calib_uvs[:, :200, :, :],
}
calib_data = {
    "points": data,
    "extrinsics": adj_extrinsics,
    "intrinsics": adj_intrinsics,
}
pickle.dump(calib_data, open(r"../tests/assets/arena_tracked_points.pkl", "wb"))