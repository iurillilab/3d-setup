from cv2.gapi import video
import numpy as np
import pandas as pd
from movement.io.load_poses import from_file
from typing import List
import cv2
from pathlib import Path
import argparse
import xarray as xr
import re
from tqdm import tqdm
import os


# extract frames for videos

def getFramesDir(src:Path)->List[int]:

    pattern = re.compile(r"frame_(\d+)\.png$")
    idx = []
    for file in src.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                number = int(match.group(1))
                idx.append(number)
    return idx
def extractFrames(video_path: Path, idxs: List[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames in video: {total_frames}")

    selected_frames = sorted(set(f for f in idxs if 0 <= f < total_frames))

    for frame_no in selected_frames:
        print(f"[INFO] Trying to extract frame: {frame_no}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret and frame is not None:
            output_path = output_dir / f"frame_{frame_no:06d}.png"
            success = cv2.imwrite(str(output_path), frame)
            print(f"[SAVED] Frame {frame_no} to {output_path}" if success else f"[ERROR] Failed to save frame {frame_no}")
        else:
            print(f"[ERROR] Could not read frame {frame_no}")

    cap.release()




# transforms labels to match new views
def permute_view_values_by_keypoint(
    df: pd.DataFrame,
    original_views: list[str],
    permutation: list[int]
) -> pd.DataFrame:
    assert len(original_views) == len(permutation), "Permutation and view list must match in length"

    # Get all keypoints by stripping the view suffix
    all_bodyparts = df.columns.get_level_values(1).unique()
    keypoints = sorted(set(
        bp.replace(f"_{view}", "")
        for bp in all_bodyparts
        for view in original_views
        if bp.endswith(f"_{view}")
    ))

    # Create backup of values before overwriting
    original_values = {}
    for keypoint in keypoints:
        for i, view in enumerate(original_views):
            full_bp = f"{keypoint}_{view}"
            for coord in ["x", "y", "likelihood"]:
                col = df.columns[
                    (df.columns.get_level_values(1) == full_bp) &
                    (df.columns.get_level_values(2) == coord)
                ][0]
                original_values[(keypoint, view, coord)] = df[col].copy()

    # Apply permutation: assign to view[i] the values from view[permutation[i]]
    for i, target_view in enumerate(original_views):
        source_view = original_views[permutation[i]]
        for keypoint in keypoints:
            for coord in ["x", "y", "likelihood"]:
                col = df.columns[
                    (df.columns.get_level_values(1) == f"{keypoint}_{target_view}") &
                    (df.columns.get_level_values(2) == coord)
                ][0]
                df[col] = original_values[(keypoint, source_view, coord)]

    return df
def preview_permutation_change(df_before: pd.DataFrame, df_after: pd.DataFrame, keypoint: str, view_from: str, view_to: str):
    for coord in ["x", "y", "likelihood"]:
        col_from = df_before.columns[
            (df_before.columns.get_level_values(1) == f"{keypoint}_{view_from}") &
            (df_before.columns.get_level_values(2) == coord)
        ][0]
        col_to = df_after.columns[
            (df_after.columns.get_level_values(1) == f"{keypoint}_{view_to}") &
            (df_after.columns.get_level_values(2) == coord)
        ][0]

        print(f"\n[{coord}] {keypoint}_{view_to} (should be copied from {view_from}):")
        print("Before:", df_before[col_to].head(5).to_list())
        print("After: ", df_after[col_to].head(5).to_list())
        print("Source:", df_before[col_from].head(5).to_list())


def rotate_central_view(df: pd.DataFrame, view_name: str, command: str, width: int, height: int) -> pd.DataFrame:
    """
    Applies rotation transform to all x/y values of the specified view (e.g., 'mirror_central') in a MultiIndex DataFrame.

    Parameters:
    - df: pandas DataFrame with MultiIndex columns (scorer, bodypart, coord)
    - view_name: str, view to apply transform to (e.g., 'mirror_central')
    - command: str, rotation command (e.g., 'transpose=1', 'vflip,hflip', etc.)
    - width, height: original frame dimensions
    """
    for col in df.columns:
        scorer, bodypart, coord = col
        if view_name in bodypart and coord in ['x', 'y']:
            x_col = df.columns[(df.columns.get_level_values(1) == bodypart) &
                               (df.columns.get_level_values(2) == 'x')][0]
            y_col = df.columns[(df.columns.get_level_values(1) == bodypart) &
                               (df.columns.get_level_values(2) == 'y')][0]

            x = df[x_col]
            y = df[y_col]

            if command == "transpose=1":
                new_x = height - y
                new_y = x
            elif command == "vflip,hflip":
                new_x = width - x
                new_y = height - y
            elif command == "transpose=2":
                new_x = y
                new_y = width - x
            else:
                raise ValueError(f"Unsupported command: {command}")

            df[x_col] = new_x
            df[y_col] = new_y

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_data", type=str, default=None, required=False)

    args = parser.parse_args()

    
    aug_data = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\tiled_views_lp")
    lp_dir = Path(r"\\wsl.localhost\Ubuntu\home\sneurobiology\Pose-app\data\dlc10v\labeled-data")


    permutation_to_view = {
           "transpose=1":[2, 0, 3, 1],
           "vflip,hflip": [3, 2, 1, 0],
           "transpose=2":[1, 3, 0, 2]}
    perm_keys = list(permutation_to_view.keys())

    H = 624
    W = 608
    views_original = ["mirror-bottom", "mirror-left", "mirror-right", "mirror-top"]

    dirs_to_process = [Path(f) for f in aug_data.iterdir()]

    for n, dir_to_process in enumerate(tqdm(dirs_to_process, desc="Processing directories")):

        # Strip "_Npermutation" to get original dir name
        original_name = re.sub(r'_\d+permutation$', '', dir_to_process.name)
        lighting_ref = lp_dir / original_name
        if not lighting_ref.exists():
            print(f"[WARNING] Original directory not found: {lighting_ref}")
            continue

        # Get frame indices
        idxs = getFramesDir(lighting_ref)

        # Load pose data
        data_path = lighting_ref / "CollectedData.csv"
        if not data_path.exists():
            print(f"[WARNING] CSV file not found: {data_path}")
            continue

        df = pd.read_csv(data_path, header=[0, 1, 2])

        # Get corresponding video
        vid_path = dir_to_process / f"{dir_to_process.name}.mp4"
        if not vid_path.exists():
            print(f"[WARNING] Video not found: {vid_path}")
            continue

        # Prepare output folder
        output_dir = dir_to_process / "extracted_frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        extractFrames(vid_path, idxs, output_dir)

        # Apply permutation based on round-robin pattern
        perm_key = perm_keys[n % len(perm_keys)]
        perm = permutation_to_view[perm_key]

        new_df = permute_view_values_by_keypoint(df.copy(), views_original, perm)
        new_df = rotate_central_view(new_df, "central", perm_key, H, W)

        # Save
        new_df.to_csv(dir_to_process / 'CollectedData.csv')
        new_df.to_hdf(dir_to_process / 'CollectedData.h5', key='df', mode='w')




    




















