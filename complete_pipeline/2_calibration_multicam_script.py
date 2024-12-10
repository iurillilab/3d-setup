#%%
# %matplotlib widget
import matplotlib
matplotlib.use('Agg')  # Configure backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

import multicam_calibration as mcc
from multicam_calibration.geometry import triangulate
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import flammkuchen as fl
from threed_utils.io import write_calibration_toml

from tqdm import tqdm, trange
import cv2
#%%

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
board_shape = (5, 7)
square_size = 12.5
# data_dir = Path("/Users/vigji/Desktop/test-anipose/cropped_calibration_vid")
data_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240724\calibration\multicam_video_2024-07-24T14_13_45_cropped_20241209165236")
output_dir = data_dir / f"mc_calibration_output_{timestamp}"
output_dir.mkdir(exist_ok=True)

video_paths = [
    f for f in data_dir.iterdir() if f.suffix == ".mp4" and "overlay" not in f.stem
]

camera_names = [p.stem.split("_")[-1].split(".avi")[0] for p in video_paths]
print(camera_names)

print("Detecting points, if not already detected...")
# detect calibration object in each video
all_calib_uvs, all_img_sizes = mcc.run_calibration_detection(
    list(map(str, video_paths)),
    mcc.detect_chessboard,
    n_workers=6,
    detection_options=dict(board_shape=board_shape, scale_factor=0.5),
)
np.save(output_dir / "all_calib_uvs.npy", all_calib_uvs)
# plot corner-match scores for each frame
fig = mcc.plot_chessboard_qc_data(video_paths)
fig.savefig(output_dir / "checkerboard_errors.png")

# optionally generate overlay videos:
overlay = False
if overlay:
    print("Generating overlay videos...")
    for p in video_paths:
        mcc.overlay_detections(p, overwrite=True)


# generate object points:
calib_objpoints = mcc.generate_chessboard_objpoints(board_shape, square_size)

fl.save(
    output_dir / "args_calibration.h5",
    dict(
        all_calib_uvs=all_calib_uvs,
        all_img_sizes=all_img_sizes,
        calib_objpoints=calib_objpoints,
    ),
)
#%%
# ================================
# Calibration
# ================================
all_extrinsics, all_intrinsics, calib_poses, spanning_tree = mcc.calibrate(
    all_calib_uvs,
    all_img_sizes,
    calib_objpoints,
    root=0,
    n_samples_for_intrinsics=100,
)

fig, shared_detections = mcc.plot_shared_detections(all_calib_uvs, spanning_tree)
fig.savefig(output_dir / "shared_detections.png")

n_cameras, n_frames, N, _ = all_calib_uvs.shape

median_error = np.zeros(n_cameras)
reprojections = np.zeros((n_cameras, n_frames, N, 2))
transformed_reprojections = np.zeros((n_cameras, n_frames, N, 2)) * np.nan
pts = mcc.embed_calib_objpoints(calib_objpoints, calib_poses)

# ================================
# Residuals
# ================================
errors_list = []
for cam in trange(n_cameras):
    reprojections[cam] = mcc.project_points(
        pts, all_extrinsics[cam], all_intrinsics[cam][0]
    )
    uvs_undistorted = mcc.undistort_points(all_calib_uvs[cam], *all_intrinsics[cam])
    valid_ixs = np.nonzero(~np.isnan(uvs_undistorted).any((-1, -2)))[0]
    for t in valid_ixs:
        H = cv2.findHomography(uvs_undistorted[t], calib_objpoints[:, :2])
        transformed_reprojections[cam, t] = cv2.perspectiveTransform(
            reprojections[cam, t][np.newaxis], H[0]
        )[0]

    errors = np.linalg.norm(
        transformed_reprojections[cam, valid_ixs] - calib_objpoints[:, :2],
        axis=-1,
    )
    median_error[cam] = np.median(errors)
    errors_arr = np.zeros(n_frames) * np.nan
    errors_arr[valid_ixs] = np.median(errors, axis=1)
    errors_list.append(errors_arr)

f, axs = plt.subplots(len(errors_list), 1, figsize=(10, 4), sharex=True, sharey=True)

for i, errors in enumerate(errors_list):
    axs[i].plot(errors + i * 20, c=f"C{i}")
f.savefig(output_dir / "residuals.png")

fig, median_error, reprojections, transformed_reprojections = mcc.plot_residuals(
    all_calib_uvs,
    all_extrinsics,
    all_intrinsics,
    calib_objpoints,
    calib_poses,
    inches_per_axis=3,
)
fig.savefig(output_dir / "first_residuals.png")


# ================================
# Bundle adjustment
# ================================
adj_extrinsics, adj_intrinsics, adj_calib_poses, use_frames, result = mcc.bundle_adjust(
    all_calib_uvs,
    all_extrinsics,
    all_intrinsics,
    calib_objpoints,
    calib_poses,
    n_frames=None,
    ftol=1e-4,
)

nan_counts = np.isnan(all_calib_uvs).sum((0, 1, 2, 3))

fig, median_error, reprojections, transformed_reprojections = mcc.plot_residuals(
    all_calib_uvs[:, use_frames],
    adj_extrinsics,
    adj_intrinsics,
    calib_objpoints,
    adj_calib_poses,
    inches_per_axis=3,
)
fig.savefig(output_dir / "refined_residuals.png")

# Write current calibration to TOML
cam_names = [Path(p).stem.split("_")[-1].split(".avi")[0] for p in video_paths]
write_calibration_toml(output_dir / "calibration_from_mc.toml", 
                      cam_names, all_img_sizes, adj_extrinsics, adj_intrinsics, result)


# ================================
# Test triangulation
# ================================

def triangulate_all_keypoints(
    calib_uvs, adj_extrinsics, adj_intrinsics, progress_bar=True
):
    all_triang = []
    progbar = tqdm if progress_bar else lambda x: x
    for i in progbar(range(calib_uvs.shape[2])):
        all_triang.append(
            triangulate(calib_uvs[:, :, i, :], adj_extrinsics, adj_intrinsics)
        )

    return np.array(all_triang)

checkboard_3d = triangulate_all_keypoints(all_calib_uvs, adj_extrinsics, adj_intrinsics)

non_nan_idxs = np.where(~np.isnan(checkboard_3d).any(axis=(0, 2)))[0]
frame_idx = non_nan_idxs[0]
checkboard_frame = checkboard_3d[:, frame_idx, :]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(checkboard_frame[:, 0], checkboard_frame[:, 1], checkboard_frame[:, 2])

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
plt.axis("equal")
plt.show()
fig.savefig(output_dir / "triangulated_frame.png")
