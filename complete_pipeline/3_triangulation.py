from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from threed_utils.io import read_calibration_toml
import matplotlib
matplotlib.use('Agg') 
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
import multiprocessing
from pipeline_params import CroppingOptions, KPDetectionOptions
from threed_utils.io import create_2d_ds, load_calibration, get_pose_files_dict, save_triangulated_ds
from threed_utils.arena_utils import load_arena_coordinates, triangulate_arena, get_arena_points_from_dataset
from threed_utils.visualization.skeleton_plots import plot_skeleton_3d, set_axes_equal
from threed_utils.anipose.movement_anipose import anipose_triangulate_ds

# triangulation function



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
            get_pose_files_dict(candidate_folder, expected_views, software)
            valid_dirs.append(candidate_folder)
        except AssertionError as e:
            print(f"Skipping {candidate_folder} because it does not have the expected views: {e}")
            continue

    return valid_dirs


def process_directory(valid_dir, calib_toml_path, triang_config_optim, expected_views, software, arena_json_path=None):
    """ Worker function to process a single directory. """
    
    ds = create_2d_ds(valid_dir, expected_views, software, max_n_frames=3000)
    threed_ds = anipose_triangulate_ds(ds, calib_toml_path, **triang_config_optim)
    
    threed_ds.attrs['fps'] = 'fps'
    threed_ds.attrs['source_file'] = 'anipose'

    save_path = save_triangulated_ds(threed_ds, valid_dir)
    
    # Create visualization with arena if provided
    if arena_json_path and arena_json_path.exists():
        try:
            # Load and triangulate arena
            arena_coordinates = load_arena_coordinates(arena_json_path)
            cam_names, _, _, _ = read_calibration_toml(calib_toml_path)
            arena_ds = triangulate_arena(arena_coordinates, calib_toml_path, cam_names)
            
            # Create visualization
            arena_points = get_arena_points_from_dataset(arena_ds, 0)
            
            # Plot first frame with arena
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot arena points
            ax.scatter(arena_points[:, 0], arena_points[:, 1], arena_points[:, 2], 
                      c='lightgray', s=50, alpha=0.8, label='Arena')
            
            # Plot mouse skeleton
            plot_skeleton_3d(threed_ds, time_idx=0, individual_idx=0, ax=ax, arena_points=None)
            ax.set_title(f'Arena and Mouse - {valid_dir.name}')
            set_axes_equal(ax)
            
            # Save visualization
            viz_path = valid_dir / f"{valid_dir.name}_arena_mouse_viz.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved arena-mouse visualization to {viz_path}")
            
        except Exception as e:
            print(f"Warning: Could not create arena visualization for {valid_dir}: {e}")
    
    return save_path  # Returning the path for tracking


def parallel_triangulation(valid_dirs, calib_toml_path, triang_config_optim, expected_views, software, num_workers=3, arena_json_path=None):
    """
    Parallelizes triangulation over multiple directories.

    :param valid_dirs: List of directories containing data.
    :param calib_toml_path: Path to calibration file.
    :param triang_config_optim: Triangulation configuration.
    :param num_workers: Number of parallel processes (default: all available cores).
    :param arena_json_path: Path to arena JSON file (optional).
    :return: List of saved file paths.
    """
    num_workers = num_workers or min(multiprocessing.cpu_count(), len(valid_dirs))

    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.starmap(process_directory, [(d, calib_toml_path, triang_config_optim, expected_views, software, arena_json_path) for d in valid_dirs]),
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
    "scale_smooth": 3,
    "scale_length": 3,
    "scale_length_weak": 0.5,
    "n_deriv_smooth": 2,
    "reproj_error_threshold": 150,
    "constraints": [['ear_lf','ear_rt'], ['nose','ear_rt'], ['nose','ear_lf'], ['tailbase', 'back_caudal'], ['back_mid', 'back_caudal'], ['back_rostral', 'back_mid']], #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    "constraints_weak": [] #[str(i), str(i+1)] for i in range(len(views_ds.coords["keypoints"])-1)],
    }
    print(f"Found {len(valid_dirs)} directories with matching views, and  calibration directories/n Proceeding with calibration")

    calibration_dir = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328/")
    
    cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path = load_calibration(calibration_dir)

    # Arena JSON path
    arena_json_path = Path("/Users/vigji/Desktop/test_3d/multicam_video_2025-05-07T10_12_11_20250528-153946.json")

    saved_files = parallel_triangulation(valid_dirs, calib_toml_path, triang_config_optim, cropping_options.expected_views, kp_detection_options.software, arena_json_path=arena_json_path)



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
