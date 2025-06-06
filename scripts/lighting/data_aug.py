import os
import shutil
from pathlib import Path
import subprocess
from xarray.backends.api import to_netcdf
from xarray.core.common import AttrAccessMixin
from movement.io.load_poses import from_file
from movement.io.save_poses import to_sleap_analysis_file
import xarray as xr
import numpy as np
import cv2
import ffmpeg
from tqdm import tqdm
import re
import argparse
from merging_videos import find_dirs_with_matching_views, buildDictFiles
import yaml

'''Script that given a dir it generates 3subdir with rotate videos tiled
Step one: generate three sub dir one per permutation

step two: rotate central view per each sub dir and convert .slp file with new

step three:

'''
import numpy as np
import xarray as xr

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
    pos = ds['position'].copy()  # shape: (time, space ['x','y'], keypoints, individuals)

    # Extract x and y positions separately
    x = pos.sel(space='x')
    y = pos.sel(space='y')

    if command == "transpose=1":
        # (x, y) -> (y, height - x)
        new_x = y
        new_y = height - x

    elif command == "vflip,hflip":
        # (x, y) -> (width - x, height - y)
        new_x = width - x
        new_y = height - y

    elif command == "transpose=2":
        # (x, y) -> (width - y, x)
        new_x = width - y
        new_y = x

    else:
        raise ValueError(f"Unsupported rotation command: {command}")

    # Stack back into a new position array
    new_position = xr.concat([new_x, new_y], dim='space')
    new_position = new_position.assign_coords(space=['x', 'y'])

    # Copy dataset and replace position
    new_ds = ds.copy()
    new_ds['position'] = new_position

    return new_ds



def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def copy_and_rename(src:Path, dst_dir:Path, new_name:str):

    dst_path = dst_dir / new_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str)
        # this is correct permutation given the rotation
    views = ["mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]

    order = {
            0: [2, 0, 3, 1],
            1: [3, 2, 1, 0],
            2: [1, 3, 0, 2]

            }


    commands = {
            0: "transpose=1",
            1: "vflip,hflip",
            2: "transpose=2"
            }

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

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    for root_dir in tqdm(dirs):

        os.makedirs(out_dir/ root_dir.name, exist_ok=True)
        dir_path = out_dir / root_dir.name
        slp_dict, vid_dict = buildDictFiles(root_dir)

        W, H = get_video_dimensions(vid_dict["central"])
        ds_central = from_file(slp_dict["central"], source_software="SLEAP")
        vids = list(vid_dict.values())[1:]

        for p in tqdm(range(3)):
            path = dir_path / f"{p}permutation"
            os.makedirs(path, exist_ok=True)
            # rotate and save central view:
            config = {
                    'rotation_command': commands[p],
                    'permutation_index': p,
                    'widht': W,
                    'height':H
                    }
            with open(path / 'transform_config.yaml', 'w') as f:
                yaml.dump(config, f)
            if not os.path.exists(path / vid_dict["central"].name):
                ffmpeg.input(vid_dict['central']).output(str(path / vid_dict["central"].name), vf=commands[p]).run()
            # rotate slp coordinates and save them:
            if  not os.path.exists(path / slp_dict["central"].name):
                shutil.copy2(slp_dict["central"], path / slp_dict["central"].name)

            for view_original, new_view in zip(views, order[p]):
                name = vid_dict[view_original].name
                if not os.path.exists(path / name):
                    vid_to_save = vid_dict[views[new_view]]
                    shutil.copy2(vid_to_save, path / name)

                slp_name = slp_dict[view_original].name
                if  not os.path.exists(path / slp_name):
                    slp_src = slp_dict[views[new_view]]
                    copy_and_rename(slp_src, path, slp_name)





