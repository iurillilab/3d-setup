from os import name
import os
from pandas import wide_to_long
import yaml
from movement.io.save_poses import to_dlc_file
from movement.io.load_poses import from_numpy
from movement.io.load_poses import from_file
from pathlib import Path
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse
import re
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing 
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


def buildDictFiles(slp_files_dir: Path) -> tuple[dict, dict]:
    '''
    Given a path(dir) it finds the .slp files and .mp4 files and builds dicts with views as key
    epx1= {top: path_video}
    '''
    assert isinstance(slp_files_dir, Path)
    slp_files = list(slp_files_dir.glob("*.slp"))
    # Windows regex
    cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)predictions\.slp$"

    # get videos:
    vid_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)\.avi.mp4$"
    vid_files = list(slp_files_dir.glob("*.mp4"))

    vid_path_dict = {re.search(vid_regex, str(f.name)).groups()[0]: f for f in vid_files}

    #mac regex
    #cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$"

    file_path_dict = {re.search(cam_regex, str(f.name)).groups()[0]: f for f in slp_files}
    # From movement.io.load_poses.from_multiview_files, split out here just to fix uppercase inconsistency bug:

    #ensure view consistency
    views_list = list(file_path_dict.keys())
    views_list_ = list(vid_path_dict.keys())
    assert views_list.sort() == views_list_.sort(), 'videos and predictions must have the same views'

    return file_path_dict, vid_path_dict

def rotate_pose_array(ds, command, width, height):
    """
    Rotate xarray Dataset of 2D poses based on an ffmpeg-style command.

    Parameters:
    - ds: xarray.Dataset with 'position' of shape (time, space, keypoints, individuals)
    - command: str, one of 'transpose=1', 'vflip,hflip', 'transpose=2'
    - width, height: original frame dimensions (in pixels)

    Returns:
    - A new xarray.Dataset with rotated coordinates
    """
    pos = ds['position'].copy()

    x = pos.sel(space='x')
    y = pos.sel(space='y')

    if command == "transpose=1":
        # 90° clockwise: (x, y) → (height - y, x)
        new_x = height - y
        new_y = x

    elif command == "vflip,hflip":
        # 180°: (x, y) → (width - x, height - y)
        new_x = width - x
        new_y = height - y

    elif command == "transpose=2":
        # 90° counterclockwise: (x, y) → (y, width - x)
        new_x = y
        new_y = width - x

    else:
        raise ValueError(f"Unsupported rotation command: {command}")

    new_position = xr.concat([new_x, new_y], dim='space')
    new_position = new_position.assign_coords(space=['x', 'y'])

    new_ds = ds.copy()
    new_ds['position'] = new_position

    return new_ds
def build2dDS(files_dict:dict, n_frames:int=None, rotate:bool=False, command=None, height=None, width=None)->xr.Dataset:


    new_coord_views = xr.DataArray(list(files_dict.keys()), dims="view")

    #idx central:
    idx_c = list(files_dict.keys()).index('central')

    dataset_list = [
        from_file(f, source_software="SLEAP")
        for f in files_dict.values()
    ]
    if rotate:
        dataset_list[idx_c] = rotate_pose_array(dataset_list[idx_c], command, height, width)
    # make coordinates labels of the keypoints axis all lowercase
    for ds in dataset_list:
        ds.coords["keypoints"] = ds.coords["keypoints"].str.lower()


    # time_slice = slice(0, 100)
    ds = xr.concat(dataset_list, dim=new_coord_views)

    ds.attrs['fps'] = 'fps'
    ds.attrs['source_file'] = 'sleap'

    return ds



def pad_to_shape(frame, target_shape):
    """Pad frame to target shape (height, width)."""
    h, w = frame.shape[:2]
    th, tw = target_shape
    top = (th - h) // 2
    bottom = th - h - top
    left = (tw - w) // 2
    right = tw - w - left
    return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
def tile_videos(video_dict, num_frames=None, layout="horizontal", output_path=None):
    """
    Tiles multiple videos into a single video with padding (streamed, not memory-heavy).

    Args:
        video_dict (dict): {view_name: video_path}
        num_frames (int or None): Number of frames to process. Defaults to min frame count across videos.
        layout (str): "horizontal", "vertical", or "grid"
        output_path (str or None): Path to save the output video

    Returns:
        first_tiled_frame: the first combined frame (e.g., for preview or plotting)
        frame_mapping: {view_name: (x_offset, y_offset, width, height, pad_top)} for coordinate shifting
    """
    view_list = ["central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]
    caps = {view: cv2.VideoCapture(str(video_dict[view])) for view in view_list}
    shapes = {}
    pad_tops = {}

    # Read first frame to get dimensions and check access
    for view, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read from video: {view}")
        shapes[view] = frame.shape[:2]  # (height, width)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    max_height = max(h for h, _ in shapes.values())
    max_width = max(w for _, w in shapes.values())
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps.values()]
    frame_counts = [fc for fc in frame_counts if fc > 0]  # Remove the -1 adjustment

    if not frame_counts:
        raise ValueError("No readable videos with valid frame counts.")

    if num_frames is None:
        num_frames = min(frame_counts)
    else:
        num_frames = min(num_frames, min(frame_counts))

    print(f"[DEBUG] Original frame counts: {frame_counts}")
    print(f"[DEBUG] Using num_frames: {num_frames}")

    # Prepare VideoWriter
    if layout == "horizontal":
        out_width = max_width * len(view_list)
        out_height = max_height
    elif layout == "vertical":
        out_width = max_width
        out_height = max_height * len(view_list)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    out = None
    if output_path:
        if os.path.exists(output_path):
            print(f"[INFO] Removing existing output video: {output_path}")
            os.remove(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 60, (out_width, out_height))

    frame_mapping = {}
    first_tiled_frame = None
    frames_written = 0

    for i in range(num_frames):
        padded_frames = []
        x_offset, y_offset = 0, 0

        for view in view_list:
            cap = caps[view]
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Video {view} ended at frame {i}, stopping processing")
                break

            h, w = shapes[view]
            pad_top = (max_height - h) // 2
            pad_bottom = max_height - h - pad_top
            pad_left = (max_width - w) // 2
            pad_right = max_width - w - pad_left

            padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            padded_frames.append(padded)

            if layout == "horizontal":
                frame_mapping[view] = (x_offset, 0, max_width, max_height, pad_top)
                x_offset += max_width
            elif layout == "vertical":
                frame_mapping[view] = (0, y_offset, max_width, max_height, pad_top)
                y_offset += max_height

        if len(padded_frames) != len(view_list):
            print(f"[WARNING] Skipping frame {i} due to incomplete views")
            continue

        if layout == "horizontal":
            tiled = np.hstack(padded_frames)
        elif layout == "vertical":
            tiled = np.vstack(padded_frames)

        if i == 0:
            first_tiled_frame = tiled.copy()

        if out:
            out.write(tiled)
            frames_written += 1

    print(f"[DEBUG] Total frames written: {frames_written}")

    if out:
        out.release()
    for cap in caps.values():
        cap.release()

    return first_tiled_frame, frame_mapping



def shift_coordinates(dataset: xr.Dataset, frame_mapping: dict)-> xr.Dataset:
    shifted = dataset.copy(deep=True)

    space_labels = list(shifted.coords["space"].values)
    x_idx = space_labels.index("x")
    y_idx = space_labels.index("y")

    for i, view in enumerate(shifted.coords["view"].values):
        x_offset, _, _, _, pad_top = frame_mapping[view]

        shifted.position[i, :, x_idx, :, :] += x_offset
        shifted.position[i, :, y_idx, :, :] += pad_top

    return shifted

def plot_keypoints_on_frame(frame, dataset: xr.Dataset, time_index=0, point_size=8):
    """
    Plot keypoints from all views on a single frame.

    Args:
        frame (np.array): Image to draw over (H, W, 3)
        dataset (xr.Dataset): Dataset with shifted coordinates
        time_index (int): Index of frame to plot
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(frame)

    # Iterate through views
    for i, view in enumerate(dataset.coords["view"].values):
        coords = dataset.position.sel(view=view).isel(time=time_index).values  # shape: (2, keypoints, individuals)
        for k in range(coords.shape[1]):
            for ind in range(coords.shape[2]):
                x = coords[0, k, ind]
                y = coords[1, k, ind]
                plt.scatter(x, y, s=point_size, label=f"{view}-{k}" if i == 0 and ind == 0 else "", alpha=0.6)

    plt.axis("off")
    plt.title(f"Tiled Frame with Overlaid Keypoints at time {time_index}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    plt.show()
def process_single_directory(args):
    root_base, perm, gen_path, output_path, num_frames = args

    perm_path = gen_path / root_base / perm
    if not perm_path.exists():
        print(f"Warning: {perm_path} does not exist, skipping.")
        return

    output_dir = output_path / f"{root_base}_{perm}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        slp_dict, video_dict = buildDictFiles(perm_path)

        # Check for optional transform config
        config_path = perm_path / "transform_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            command = config['rotation_command']
            width = config['widht']
            height = config['height']
            ds = build2dDS(slp_dict, rotate=True, command=command, height=height, width=width)
        else:
            ds = build2dDS(slp_dict)

        # Save tiled video
        tiled_output_path = output_dir / f"{root_base}_{perm}.mp4"
        tiled_frames, mapping = tile_videos(
            video_dict,
            num_frames=num_frames,
            layout="horizontal",
            output_path=tiled_output_path
        )
        print(f"[{root_base}_{perm}] Tiled video saved to {tiled_output_path}")

        # Save predictions
        shifted_ds = shift_coordinates(ds, mapping)
        pred_output_path = output_dir / f"predictions_{root_base}_{perm}.h5"
        shifted_ds.to_netcdf(pred_output_path)
        print(f"[{root_base}_{perm}] Predictions saved to {pred_output_path}")

    except Exception as e:
        print(f"Error processing {root_base}/{perm}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process and tile videos.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to process.")
    parser.add_argument("--output_path", type=str, default="tiled_output", help="Directory to save outputs.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    NUM_FRAMES = args.num_frames
    OUTPUT_PATH = Path(args.output_path)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    GEN_PATH = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\augmented_views")
    PERMUTATIONS = ["0permutation", "1permutation", "2permutation"]

    task_list = []
    for root_base in os.listdir(GEN_PATH):
        if not (GEN_PATH / root_base).is_dir():
            continue
        for perm in PERMUTATIONS:
            task_list.append((root_base, perm, GEN_PATH, OUTPUT_PATH, NUM_FRAMES))

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_directory, task_list), total=len(task_list), desc="Processing all permutations"))

if __name__ == "__main__":

    main()

    # parser = argparse.ArgumentParser(description="Process and tile videos.")
    # parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to process.")
    # parser.add_argument("--output_path", type=str, default="tiled_output", help="Directory to save merged videos and predictions.")
    # args = parser.parse_args()

    # NUM_FRAMES = args.num_frames
    # OUTPUT_PATH = Path(args.output_path)
    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # gen_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\augmented_views")
    # permutations = ["0permutation", "1permutation", "2permutation"]

    # for root_base in tqdm(os.listdir(gen_path), desc="Processing root dirs"):
    #     root_base_path = gen_path / root_base
    #     if not root_base_path.is_dir():
    #         continue

    #     for perm in permutations:
    #         perm_path = root_base_path / perm
    #         if not perm_path.exists():
    #             print(f"Warning: {perm_path} does not exist, skipping.")
    #             continue

    #         # Define output directory based on root_base and permutation
    #         output_dir = OUTPUT_PATH / f"{root_base}_{perm}"
    #         output_dir.mkdir(parents=True, exist_ok=True)

    #         slp_dict, video_dict = buildDictFiles(perm_path)

    #         # Check for optional transform config
    #         config_path = perm_path / "transform_config.yaml"
    #         if config_path.exists():
    #             with open(config_path, 'r') as f:
    #                 config = yaml.safe_load(f)
    #             command = config['rotation_command']
    #             width = config['widht']
    #             height = config['height']
    #             ds = build2dDS(slp_dict, rotate=True, command=command, height=height, width=width)
    #         else:
    #             ds = build2dDS(slp_dict)

    #         # Save tiled video
    #         tiled_output_path = output_dir / f"{root_base}_{perm}.mp4"
    #         tiled_frames, mapping = tile_videos(
    #             video_dict,
    #             num_frames=NUM_FRAMES,
    #             layout="horizontal",
    #             output_path=tiled_output_path
    #         )
    #         print(f"Tiled video saved to {tiled_output_path}")

    #         # Save predictions
    #         shifted_ds = shift_coordinates(ds, mapping)
    #         pred_output_path = output_dir / f"predictions_{root_base}_{perm}.h5"
    #         shifted_ds.to_netcdf(pred_output_path)
    #         print(f"Saved shifted dataset to {pred_output_path}")

        # plot_keypoints_on_frame(tiled_frames[0], shifted_ds, time_index=0)

