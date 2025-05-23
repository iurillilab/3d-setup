from os import name
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
    Tiles multiple videos into a single video with padding (no resizing).

    Args:
        video_dict (dict): {view_name: video_path}
        num_frames (int): Max number of frames (or None to use min across all).
        layout (str): "horizontal", "vertical", or "grid"
        output_path (str or None): Path to save tiled video

    Returns:
        tiled_frames (list): List of combined frames
        frame_mapping (dict): {view_name: (x_offset, y_offset, width, height, pad_top)}
    """
    caps = {view: cv2.VideoCapture(path) for view, path in video_dict.items()}
    first_frames = {}
    shapes = {}

    # Read first frame to get sizes
    for view, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Couldn't read from video: {view}")
        shapes[view] = frame.shape[:2]  # (height, width)
        first_frames[view] = [frame]

    # Find max height and width for padding
    max_height = max(h for h, w in shapes.values())
    max_width = max(w for h, w in shapes.values())

    # Determine number of frames
    if num_frames is None:
        num_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps.values()))

    # Read remaining frames
    for i in range(1, num_frames):
        for view, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Video {view} ended early at frame {i}")
            first_frames[view].append(frame)

    # Pad and store all frames
    padded_frames = {view: [] for view in video_dict}
    pad_tops = {}

    for view, frames in first_frames.items():
        h, w = shapes[view]
        pad_top = (max_height - h) // 2
        pad_tops[view] = pad_top
        for frame in frames:
            padded = pad_to_shape(frame, (max_height, max_width))
            padded_frames[view].append(padded)

    # Build tiled frames
    tiled_frames = []
    frame_mapping = {}
    view_list = list(video_dict.keys())

    for i in range(num_frames):
        current_frames = [padded_frames[view][i] for view in view_list]

        if layout == "horizontal":
            tiled = np.hstack(current_frames)
            x = 0
            for view in view_list:
                frame_mapping[view] = (x, 0, max_width, max_height, pad_tops[view])
                x += max_width

        elif layout == "vertical":
            tiled = np.vstack(current_frames)
            y = 0
            for view in view_list:
                frame_mapping[view] = (0, y, max_width, max_height, pad_tops[view])
                y += max_height

        elif layout == "grid":
            rows = int(np.ceil(np.sqrt(len(view_list))))
            cols = int(np.ceil(len(view_list) / rows))
            blank = np.zeros((max_height, max_width, 3), dtype=np.uint8)

            grid = []
            idx = 0
            for r in range(rows):
                row = []
                for c in range(cols):
                    if idx < len(view_list):
                        frame = current_frames[idx]
                        view = view_list[idx]
                        x_offset = c * max_width
                        y_offset = r * max_height
                        frame_mapping[view] = (x_offset, y_offset, max_width, max_height, pad_tops[view])
                        row.append(frame)
                        idx += 1
                    else:
                        row.append(blank)
                grid.append(np.hstack(row))
            tiled = np.vstack(grid)

        else:
            raise ValueError(f"Unsupported layout: {layout}")

        tiled_frames.append(tiled)

    # Save output if requested
    if output_path:
        h, w, _ = tiled_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
        for frame in tiled_frames:
            out.write(frame)
        out.release()

    for cap in caps.values():
        cap.release()

    return tiled_frames, frame_mapping

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
#TODO Rotate central view and permute laterla view to obtain 4 videos

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and tile videos.")
    parser.add_argument("--num_frames", type=int, default=5000, help="Number of frames to process.")
    parser.add_argument("--output_path", type=str, default="tiled_output.mp4", help="Path to save the tiled video.")


    args = parser.parse_args()
    NUM_FRAMES = args.num_frames
    root_dirs = [r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240722\M1\101552\multicam_video_2024-07-22T10_19_22_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240801\M2\111448\multicam_video_2024-08-01T12_00_10_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240801\M8\163429\multicam_video_2024-08-01T17_06_27_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240805\M4\144038\multicam_video_2024-08-05T15_05_00_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240803\M4\140337\multicam_video_2024-08-03T14_32_11_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240805\M3\124846\multicam_video_2024-08-05T14_40_38_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240805\M2\111721\multicam_video_2024-08-05T11_38_40_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240805\M5\150500\multicam_video_2024-08-05T15_22_07_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240801\M6\153300\multicam_video_2024-08-01T15_59_25_cropped_20250325101012",
                 r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240724\112233\multicam_video_2024-07-24T11_37_02_cropped_20250325101012"
                 ]
    dirs = [Path(p) for p in root_dirs]
    gen_path = Path("/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012")
    dirs = [gen_path / "0permutation", gen_path / "1permutation", gen_path / "2permutation"]
    output_path = Path(args.output_path)



    # "D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project"

    for root_dir in tqdm(dirs, desc="Processing directories"):


        # get config parms :
       config_path = root_dir / "transform_config.yaml"
       if "permutation" in root_dir.name:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        command = config['rotation_command']
        width = config['widht']
        height = config['height']




        slp_dict, video_dict = buildDictFiles(Path(root_dir))


        tiled_frames, mapping = tile_videos(
            video_dict,
            num_frames=NUM_FRAMES,
            layout="horizontal",  # or "horizontal"
            output_path=output_path / f"{root_dir.name}.mp4")
        print(f"Tiled video saved to {output_path / 'tiled_output.mp4'}")
        if "permutation" in root_dir.name:
            ds = build2dDS(slp_dict, rotate=True, command=command, height=height, width=width)

        else:
            ds = build2dDS(slp_dict)

        shifted_ds = shift_coordinates(ds, mapping)
        save_path = output_path / f"predictions_{root_dir.name}.h5"
        shifted_ds.to_netcdf(save_path)
        print(f"Saved shifted dataset to {save_path}")



        plot_keypoints_on_frame(tiled_frames[0], shifted_ds, time_index=0)


