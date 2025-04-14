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
import cv2
import multiprocessing



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
                    return f
        current = current.parent
    return None

def find_dirs_with_matching_views(root_dir: Path, expected_views: set, recompute=True) -> list[Path]:
    """
    Find directories containing exactly 5 SLP files with matching camera views.
    """
    valid_dirs = []

    # Windows regex
    #cam_regex = r"^[A-Za-z]:\\(?:[^\\]+\\)*multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$"
    # cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)predictions\.slp$"

    all_candidate_folders = [f for f in root_dir.rglob("multicam_video_*_cropped_*") if f.is_dir()]
    parent_dict = {folder.parent: [] for folder in all_candidate_folders}
    
    for candidate_folder in all_candidate_folders:
        parent_dict[candidate_folder.parent].append(candidate_folder)
    
    last_folders = [sorted(folders)[-1] for folders in parent_dict.values()]

    #Unix regex
    #cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$"  # Updated regex

    # Recursively iterate through all directories
    # for directory in root_dir.rglob('*'):
    for directory in last_folders:    
        #if not directory.is_dir():
        #    continue

        if "calibration" in [parent.name.lower() for parent in directory.parents]:
            continue
        # Get all SLP files in the current directory
        slp_files = list(directory.glob('*.slp'))

        if not recompute and len(list(directory.glob("*triangulated_points_*.h5"))) > 0:
            continue

        # Extract camera views from filenames
        # current_views = set()
        #for f in slp_files:
        #    match = re.search(cam_regex, f.name)
        #    if match:
        #        camera_name = match.group(1)  # Extract camera view name
        #        current_views.add(camera_name)
        #    else:
        #        continue


        # If we found exactly 5 matching views and they match the expected views
        # if len(current_views) == 5 and current_views == expected_views:
        if len(slp_files) == 5:
            valid_dirs.append(directory)
    valid_dirs.reverse() # to avoid possible error 
    return valid_dirs

def create_2d_ds(slp_files_dir: Path):
    slp_files = list(slp_files_dir.glob("*.slp"))
    # Windows regex
    cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)predictions\.slp$"



    #mac regex
    #cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$" 


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


    # time_slice = slice(0, 1000)
    ds = xr.concat(dataset_list, dim=new_coord_views)

    bodyparts = list(ds.coords["keypoints"].values)

    print(bodyparts)

    print(ds.position.shape, ds.confidence.shape, bodyparts)

    ds.attrs['fps'] = 'fps'
    ds.attrs['source_file'] = 'sleap'

    return ds


def generate_calibration_data(calibration_dir):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    board_shape = (5, 7)
    square_size = 12.5
    data_dir = calibration_dir
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
    
    return output_dir

def process_directory(valid_dir, calib_toml_path, triang_config_optim):
    """ Worker function to process a single directory. """
    print(valid_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    ds = create_2d_ds(valid_dir)
    _3d_ds = anipose_triangulate_ds(ds, calib_toml_path, **triang_config_optim)
    
    _3d_ds.attrs['fps'] = 'fps'
    _3d_ds.attrs['source_file'] = 'anipose'

    # Save the triangulated points using the directory name
    save_path = valid_dir / f"{valid_dir.name}_triangulated_points_{timestamp}.h5"
    _3d_ds.to_netcdf(save_path)

    return save_path  # Returning the path for tracking

def parallel_triangulation(valid_dirs, calib_toml_path, triang_config_optim, num_workers=3):
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
            pool.starmap(process_directory, [(d, calib_toml_path, triang_config_optim) for d in valid_dirs]),
            total=len(valid_dirs),
            desc="Triangulating directories"
        ))

    return results  # List of saved file paths

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Triangulate all files in a directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    expected_views = {'mirror-bottom', 'mirror-left', 'mirror-top', 'central', 'mirror-right'}
    valid_dirs = find_dirs_with_matching_views(data_dir, expected_views, recompute=False)
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
    "constraints": [['lear','rear'], ['nose','rear'], ['nose','lear'], ['tailbase', 'upperback'], ['uppermid', 'upperback'], ['upperforward', 'uppermid']], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
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
    calibration_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240722\calibration\multicam_video_2024-07-24T14_13_45_cropped_20241209165236")
    if not calibration_dir:
        calibration_data_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240722\calibration\multicam_video_2024-07-24T14_13_45_cropped_20241209165236")
        calibration_dir = generate_calibration_data(calibration_data_dir) 
    
    cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calibration_dir)

    saved_files = parallel_triangulation(valid_dirs, calib_toml_path, triang_config_optim)



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
