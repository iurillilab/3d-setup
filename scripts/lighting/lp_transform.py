import numpy as np
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
    vid_paths = [vid.name for vid in (aug_data / "tiled").iterdir() if ".mp4" in vid.name]


    output_dir = aug_data / "tiled/perm1"

    extractFrames(aug_data / "tiled" / vid_paths[0], idxs, output_dir)





