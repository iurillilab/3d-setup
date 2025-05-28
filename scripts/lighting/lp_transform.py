import numpy as np
import pandas as pd
from movement.io.load_poses import from_file
from typing import List
import cv2
from pathlib import Path
import argparse
import xarray as xr
import re



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
    parser.add_argument("--lp_dir", type=str)

    args = parser.parse_args()

    lp_dir = Path(args.lp_dir)
    aug_data = Path("/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012")

    lp_dir = lp_dir / aug_data.name
    idxs = getFramesDir(lp_dir)

    # extract frames and save them in dir:
    # vid_paths = [vid.name for vid in (aug_data / "tiled").iterdir() if ".mp4" in vid.name]
    #
    #
    # output_dir = aug_data / "tiled/perm1"
    #
    # extractFrames(aug_data / "tiled" / vid_paths[0], idxs, output_dir)
    #
    # convert the coordinate and save them:
    data_path = lp_dir / "CollectedData.csv"
    df = pd.read_csv(data_path, header=[0, 1, 2])
    # now we operate on the csv to rotate the central view and permute
    #the other views
    #TODO check that it works with the same W, H from original









