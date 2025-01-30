from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multicam_calibration.geometry as mcc_geom
import pandas as pd
import xarray as xr
from movement.io.load_poses import from_numpy
from threed_utils.io import movement_ds_from_anipose_triangulation_df, read_calibration_toml
from threed_utils.anipose.triangulate import CameraGroup, triangulate_core
import argparse
import re
from movement.io.load_poses import from_file



def load_calibration(calibration_dir: Path):


    calibration_paths = sorted(calibration_dir.glob("mc_calibration_output_*"))
    last_calibration_path = calibration_paths[-1]

    all_calib_uvs = np.load(last_calibration_path / "all_calib_uvs.npy")
    calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
    cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)

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
    # TODO: add sorting dimension  form the toml file to 

    triang_df = triangulate_core(config, 
                 positions, 
                 scores, 
                 views_ds.coords["keypoints"].values, 
                 cgroup, 
                 )

    return movement_ds_from_anipose_triangulation_df(triang_df)



def find_closest_calibration_dir(dir_path: Path) -> Path | None:
    """Find the closest calibration directory by walking up the directory tree."""
    current = dir_path
    while current != current.parent:  # Stop at root directory
        calib_dir = current / 'calibration'
        if calib_dir.exists() and calib_dir.is_dir():
            return calib_dir
        current = current.parent
    return None

def find_dirs_with_matching_views(root_dir: Path, expected_views: set) -> list[Path]:
    """
    Find directories containing exactly 5 SLP files with matching camera views.
    """
    valid_dirs = []
    cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$"  # Updated regex

    # Recursively iterate through all directories
    for directory in root_dir.rglob('*'):
        if not directory.is_dir():
            continue
        if directory.name == "calibration":
            continue
        # Get all SLP files in the current directory
        slp_files = list(directory.glob('*.slp'))
        
        # Skip if not exactly 5 files
        if len(slp_files) != 5:
            continue

        # Extract camera views from filenames
        current_views = set()
        for f in slp_files:
            match = re.search(cam_regex, f.name)
            if match:
                camera_name = match.group(1)  # Extract camera view name
                current_views.add(camera_name)
            else:
                print(f"Warning: Filename {f.name} doesn't match expected pattern")
                break

        # If we found exactly 5 matching views and they match the expected views
        if len(current_views) == 5 and current_views == expected_views:
            valid_dirs.append(directory)

    return valid_dirs

def create_2d_ds(slp_files_dir: Path):
    slp_files = list(slp_files_dir.glob("*.slp"))

    cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$" 


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

    print(bodyparts)

    print(ds.position.shape, ds.confidence.shape, bodyparts)

    ds.attrs['fps'] = 'fps'
    ds.attrs['source_file'] = 'sleap'

    return ds



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Triangulate all files in a directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    expected_views = {'mirror-bottom', 'mirror-left', 'mirror-top', 'central', 'mirror-right'}
    valid_dirs = find_dirs_with_matching_views(data_dir, expected_views)
    calib_dirs = [find_closest_calibration_dir(dir) for dir in valid_dirs]

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
    "constraints": [['lear','rear'], ['nose','rear'], ['nose','lear'], ['tailbase', 'upperback']], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    "constraints_weak": [] #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    }


    for calib_dir in calib_dirs:
        cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calib_dir)
    
    for dir in valid_dirs:
        ds = create_2d_ds(dir)

    for valid_dir in tqdm(valid_dirs, desc="Triangulating directories"):
        ds = create_2d_ds(valid_dir)
        _3d_ds = anipose_triangulate_ds(ds, calib_toml_path, **triang_config_optim)
        _3d_ds.attrs['fps'] = 'fps'
        _3d_ds.attrs['source_file'] = 'anipose'
        # Save the triangulated points using the directory name
        save_path = valid_dir / f"{valid_dir.name}_triangulated_points.h5"
        _3d_ds.to_netcdf(save_path)

    




