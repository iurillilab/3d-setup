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
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import flammkuchen as fl
from threed_utils.io import write_calibration_toml
from tqdm import tqdm, trange
import cv2
import multiprocessing
from pipeline_params import CroppingOptions, KPDetectionOptions
from movement.io.load_poses import from_multiview_files



def load_calibration(calibration_dir: Path):
    calibration_paths = sorted(calibration_dir.glob("mc_calibration_output_*"))
    last_calibration_path = calibration_paths[-1]

    calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
    cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)
    print("Got calibration for the following cameras: ", cam_names)
    return cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path

# triangulation function
def anipose_triangulate_ds(views_ds, calib_toml_path, **config_kwargs):
    triang_config = config_kwargs
    config = dict(triangulation=triang_config)

    calib_fname = str(calib_toml_path)
    cgroup = CameraGroup.load(calib_fname)
    # read toml file and use the views to order the dimenensions of the views_ds, so thne you are sure that when you will do the back projeciton thsoe are the same order of the matrices.

    individual_name = views_ds.coords["individuals"][0]
    reshaped_ds = views_ds.sel(individuals=individual_name).transpose("view", "time", "keypoints", "space")
    # sort over view axis using the view ordring
    positions = reshaped_ds.position.values
    scores = reshaped_ds.confidence.values

    triang_df = triangulate_core(config, 
                 positions, 
                 scores, 
                 views_ds.coords["keypoints"].values, 
                 cgroup, 
                 )

    return movement_ds_from_anipose_triangulation_df(triang_df)



def find_closest_calibration_dir(dir_path: Path) -> Path | None:
    """Find the closest calibration directory by walking up the directory tree."""
    reg = r"_cropped_"
    current = dir_path
    while current != current.parent:  # Stop at root directory
        calib_dir = current / 'calibration'
        if calib_dir.exists() and calib_dir.is_dir():
            l = [f for f in calib_dir.iterdir()]
            for f in l:
                if re.search(reg, f.name):
                    print(f"Found calibration directory {f} for {dir_path}")
                    return f
        current = current.parent
    raise ValueError(f"No calibration directory found in {dir_path}")


def get_view_from_filename_deeplabcut(filename: str) -> str:
    return filename.split("DLC")[0].split("_")[-1]


def get_view_from_filename_sleap(filename: str) -> str:
    raise NotImplementedError("SLEAP loading not supported yet")


def get_tracked_files_dict(dir_path: Path, expected_views: tuple[str], software: str) -> dict[str, Path]:
    suffix = None
    parsing_function = None
    if software == "DeepLabCut":
        suffix = "h5"
        parsing_function = get_view_from_filename_deeplabcut
    elif software == "SLEAP":
        suffix = "slp"
        parsing_function = get_view_from_filename_sleap
    else:
        raise ValueError(f"Non supported software: {software}")
    tracked_files = sorted(dir_path.glob(f"*{suffix}"))

    file_path_dict = {parsing_function(f.stem): f for f in tracked_files}
    file_path_dict = dict(sorted(file_path_dict.items()))
    keys_tuple = tuple(file_path_dict.keys())
    assert keys_tuple == expected_views, f"Expected views {expected_views}, got {keys_tuple}"
    return file_path_dict


def find_dirs_with_matching_views(root_dir: Path, expected_views: set, crop_folder_pattern: str, software: str) -> list[Path]:
    """
    Find directories containing exactly 5 SLP files with matching camera views.
    """
    valid_dirs = []

    all_candidate_folders = [f for f in root_dir.glob(f"*{crop_folder_pattern}*") if f.is_dir()]
    assert len(all_candidate_folders) > 0, f"No candidate folders found in {root_dir} with pattern {crop_folder_pattern}"

    valid_dirs = []
    for candidate_folder in all_candidate_folders:
        try:
            get_tracked_files_dict(candidate_folder, expected_views, software)
            valid_dirs.append(candidate_folder)
        except AssertionError as e:
            print(f"Skipping {candidate_folder} because it does not have the expected views: {e}")
            continue

    return valid_dirs

def create_2d_ds(slp_files_dir: Path, expected_views: tuple[str], software: str, max_n_frames: int = 300):
    file_path_dict = get_tracked_files_dict(slp_files_dir, expected_views, software)

    ds = from_multiview_files(file_path_dict, source_software=software)

    if max_n_frames is not None:
        ds = ds.isel(time=slice(0, max_n_frames))

    # print(ds.position.shape, ds.confidence.shape, ds.coords["keypoints"].values)

    return ds

def process_directory(valid_dir, calib_toml_path, triang_config_optim, expected_views, software):
    """ Worker function to process a single directory. """
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    ds = create_2d_ds(valid_dir, expected_views, software)
    threed_ds = anipose_triangulate_ds(ds, calib_toml_path, **triang_config_optim)
    
    threed_ds.attrs['fps'] = 'fps'
    threed_ds.attrs['source_file'] = 'anipose'

    # Save the triangulated points using the directory name
    save_path = valid_dir / f"{valid_dir.name}_triangulated_points_{timestamp}.h5"

    threed_ds.to_netcdf(save_path)

    return save_path  # Returning the path for tracking


def parallel_triangulation(valid_dirs, calib_toml_path, triang_config_optim, expected_views, software, num_workers=3):
    """
    Parallelizes triangulation over multiple directories.

    :param valid_dirs: List of directories containing data.
    :param calib_toml_path: Path to calibration file.
    :param triang_config_optim: Triangulation configuration.
    :param num_workers: Number of parallel processes (default: all available cores).
    :return: List of saved file paths.
    """
    num_workers = num_workers or min(multiprocessing.cpu_count(), len(valid_dirs))

    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.starmap(process_directory, [(d, calib_toml_path, triang_config_optim, expected_views, software) for d in valid_dirs]),
            total=len(valid_dirs),
            desc="Triangulating directories"
        ))

    return results  # List of saved file paths

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description="Triangulate all files in a directory")
    # parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the data")
    # args = parser.parse_args()
    cropping_options = CroppingOptions()
    data_dir = Path("/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050") #  Path(args.data_dir)
    kp_detection_options = KPDetectionOptions()
    expected_views = {'mirror-bottom', 'mirror-left', 'mirror-top', 'central', 'mirror-right'}
    valid_dirs = find_dirs_with_matching_views(data_dir, cropping_options.expected_views, cropping_options.crop_folder_pattern, kp_detection_options.software)
    #calib_dirs = [find_closest_calibration_dir(dir) for dir in valid_dirs]
    toml_files = []

    triang_config_optim = {
    "ransac": True,
    "optim": True,
    "optim_chunking": True,
    "optim_chunking_size": 100,
    "score_threshold": 0.7,
    "scale_smooth": 1,
    "scale_length": 3,
    "scale_length_weak": 0.5,
    "n_deriv_smooth": 2,
    "reproj_error_threshold": 150,
    "constraints": [['ear_lf','ear_rt'], ['nose','ear_rt'], ['nose','ear_lf'], ['tailbase', 'back_caudal'], ['back_mid', 'back_caudal'], ['back_rostral', 'back_mid']], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    "constraints_weak": [] #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    }
    print(f"Found {len(valid_dirs)} directories with matching views, and  calibration directories/n Proceeding with calibration")

    # calibrations_set = set(calib_dirs)
    # for dir in calibrations_set:
    #     generate_calibration_data(Path(dir))
    # for calib_dir in calib_dirs:
    #     cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calib_dir)
    #     toml_files.append(calib_toml_path)
    # print("calibrations generated and loaded successfully")

    # just using a signle calibration dir for now
    # calibration_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240722\calibration\multicam_video_2024-07-24T14_13_45_cropped_20241209165236")
    calibration_dir = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328/")
    
    cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calibration_dir)

    saved_files = parallel_triangulation(valid_dirs, calib_toml_path, triang_config_optim, cropping_options.expected_views, kp_detection_options.software)



    # for valid_dir in tqdm(valid_dirs, desc="Triangulating directories"):
    #     print(valid_dir)
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     ds = create_2d_ds(valid_dir)
    #     _3d_ds = anipose_triangulate_ds(ds, calib_toml_path, **triang_config_optim)
    #     _3d_ds.attrs['fps'] = 'fps'
    #     _3d_ds.attrs['source_file'] = 'anipose'
    #     # Save the triangulated points using the directory name
    #     save_path = valid_dir / f"{valid_dir.name}_triangulated_points_{timestamp}.h5"
    #     _3d_ds.to_netcdf(save_path)

    # sovle issue with missing calibration dir during first day, (just copy a calibration dir from another day)
