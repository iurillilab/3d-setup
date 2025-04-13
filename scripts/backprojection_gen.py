#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import xarray as xr
from threed_utils.io import movement_ds_from_anipose_triangulation_df, read_calibration_toml
from threed_utils.anipose.triangulate import CameraGroup, triangulate_core
import argparse
import re
from movement.io.load_poses import from_file
import matplotlib
matplotlib.use('Agg') 
import multicam_calibration as mcc
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import flammkuchen as fl
from threed_utils.io import write_calibration_toml
from tqdm import tqdm, trange

from multicam_calibration.geometry import project_points
#%%

def load_calibration(calibration_dir: Path):


    calibration_paths = sorted(calibration_dir.glob("mc_calibration_output_*"))
    last_calibration_path = calibration_paths[-1]

    all_calib_uvs = np.load(last_calibration_path / "all_calib_uvs.npy")
    calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
    cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)

    return cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path

def find_dirs_with_matching_views(root_dir: Path) -> list[Path]:
    """
    Find directories containing exactly 5 SLP files with matching camera views.
    """
    valid_dirs = []

    all_candidate_folders = [f for f in root_dir.rglob("multicam_video_*_cropped_*") if f.is_dir()]
    parent_dict = {folder.parent: [] for folder in all_candidate_folders}
    
    for candidate_folder in all_candidate_folders:
        parent_dict[candidate_folder.parent].append(candidate_folder)
    
    last_folders = [sorted(folders)[-1] for folders in parent_dict.values()]


    for directory in last_folders:    
        #if not directory.is_dir():
        #    continue

        if "calibration" in [parent.name.lower() for parent in directory.parents]:
            continue
        # Get all SLP files in the current directory
        slp_files = list(directory.glob('*.slp'))

        if  len(list(directory.glob("*triangulated_points_*.h5"))) < 0:
            continue
        
        for h5 in directory.glob("*triangulated_points_*.h5"):
            if not h5.is_file():
                continue
            valid_dirs.append(h5)
    valid_dirs.reverse() # to avoid possible error 
    return valid_dirs


def backproject_triangulated_points(xarray_ds, extrinsics, intrinsics, cam_names):
    """
    Back-project 3D triangulated points to 2D camera coordinates.

    Args:
        xarray_ds (xarray.Dataset): Dataset with dims (time, space, keypoints, individuals)
        extrinsics (dict): Dict of extrinsic vectors (6,) per camera
        intrinsics (dict): Dict of tuples (camera_matrix, distortion) per camera
        cam_names (list): List of camera names (keys for extrinsics/intrinsics)

    Returns:
        np.ndarray: shape (n_cameras, n_frames, n_keypoints, 2)
    """

    n_frames = xarray_ds.time.size
    n_keypoints = xarray_ds.keypoints.size
    n_cameras = len(cam_names)

    # Reorder to (n_frames, n_keypoints, 3)
    positions = np.moveaxis(xarray_ds.position.values[:, :, :, 0], 1, 2)  # shape: (frames, keypoints, 3)

    # Prepare output
    reprojections = np.zeros((n_cameras, n_frames, n_keypoints, 2))

    for i, cam in enumerate(tqdm(cam_names, desc="Backprojecting points")):
        extr = extrinsics[i]                     # (6,)
        cam_mat, dist_coefs = intrinsics[i]      # (3, 3), (k1, k2, ...)
        
        reprojections[i] = project_points(positions, extr, cam_mat, dist_coefs)
    

    return reprojections




if __name__ == "__main__":

    main_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting")
    save_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting/backprojection")
    save_dir.mkdir(parents=True, exist_ok=True)


    expected_views = {'mirror-bottom', 'mirror-left', 'mirror-top', 'central', 'mirror-right'}

    # step one is to load calibration matrix
    calibration_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240722\calibration\multicam_video_2024-07-24T14_13_45_cropped_20241209165236")
    cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calibration_dir)


    # step two is to load the triangulation data path:
    h5_files = find_dirs_with_matching_views(main_dir)
    print("Found files:", len(h5_files))

    for h5_file in tqdm(h5_files, desc="Backprojecting files"):
        if 'reprojection' in h5_file.name:
            continue
        print("Processing file:", h5_file)
        # Load the triangulated points
        xarray_ds = xr.open_dataset(h5_file)
        

        #extract confidence from the data
        # Expand confidence to shape (n_cameras, n_frames, n_keypoints)
        confidence = xarray_ds.confidence.values  # shape: (n_frames, n_keypoints)
        confidence_expanded = np.repeat(confidence[None, :, :], len(cam_names), axis=0)

        # Make sure shape is exactly (n_cameras, n_frames, n_keypoints)
        confidence_expanded = confidence_expanded.reshape(len(cam_names), confidence.shape[0], confidence.shape[1])


        # Back-project the triangulated points to 2D camera coordinates
        reprojections = backproject_triangulated_points(xarray_ds, extrinsics, intrinsics, cam_names)

        # save to xarray 


        # Save to xarray
        reprojections_ds = xr.Dataset(
            {
                "reprojections": (("camera", "time", "keypoints", "xy"), reprojections),
                "confidence": (("camera", "time", "keypoints"), confidence_expanded),
            },
            coords={
                "camera": cam_names,
                "time": xarray_ds.time.values,
                "keypoints": xarray_ds.keypoints.values,
                "xy": ["x", "y"],
            },
        )


        #save to h5 file
        save_path = h5_file.parent / f"reprojection_{h5_file.name}"
        reprojections_ds.to_netcdf(save_path)

                

        # save reprojection into new dir 







