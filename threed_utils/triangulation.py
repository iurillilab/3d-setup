from pathlib import Path

import multicam_calibration as mcc
import numpy as np
from tqdm import tqdm


def triangulate_all_keypoints(
    calib_uvs, adj_extrinsics, adj_intrinsics, progress_bar=True
):
    """
    It triangulates all the keypoints from all the cameras in the calibration set.
    Parameters:
    calib_uvs: numpy array of shape (n_cameras, n_frames, keypoints,  2)
    adj_extrinsic: numpy array of shape (n_cameras, 6)
    adj_intrinsics: list of length n_cameras, each element: numpy array of shape (3, 3), dist_coef (n_camers, 1)

    Returns:
    all_triang: numpy array of shape (keypoings, frames, 3)
    """

    all_triang = []
    progbar = tqdm if progress_bar else lambda x: x
    for i in progbar(range(calib_uvs.shape[2])):
        all_triang.append(
            mcc.triangulate(calib_uvs[:, :, i, :], adj_extrinsics, adj_intrinsics)
        )
    return np.array(all_triang)


def back_project_points(three_d_points, adj_extrinsics, adj_intrinsics, cam_names):
    """
    It back projects the 3D points to all the cameras in the calibration set.
    Parameters:
    three_d_points: numpy array of shape (keypoints, frames, 3)
    adj_extrinsic: numpy array of shape (n_cameras, 6)
    adj_intrinsics: list of length n_cameras, each element: numpy array of shape (3, 3), dist_coef (n_camers, 1)
    cam_names: list of length n_cameras

    Returns:
    all_back_proj: dictionary of length n_cameras, each element: numpy array of shape (keypoints, frames, 2)
    """

    all_back_proj = {}
    for extrinsic, intrinsic in zip(adj_extrinsics, adj_intrinsics):
        all_back_proj[cam_names.pop(0)] = mcc.project_points(
            three_d_points, extrinsic, intrinsic
        )
    return all_back_proj
