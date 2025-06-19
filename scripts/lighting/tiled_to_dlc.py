#%%
import shutil
from pathlib import Path
from datetime import datetime
import xarray as xr
import yaml
import pandas as pd
#%%

# === Reuse your own config generator ===
def generate_config(scorer, project_path, video_paths, bodyparts):
    date = datetime.now().strftime("%b%d")
    video_sets = {str(v): {"crop": "0, 1108, 0, 752"} for v in video_paths}
    grouped_bodyparts = {}
    for bp in bodyparts:
        prefix = bp.split('_')[0]
        grouped_bodyparts.setdefault(prefix, []).append(bp)

    skeleton = []  # add logic if needed

    config = {
        "Task": "mouseface",
        "scorer": scorer,
        "date": date,
        "multianimalproject": False,
        "identity": None,
        "project_path": str(project_path),
        "video_sets": video_sets,
        "bodyparts": bodyparts,
        "start": 0,
        "stop": 1,
        "numframes2pick": 20,
        "skeleton": skeleton,
        "skeleton_color": "black",
        "pcutoff": 0.6,
        "dotsize": 12,
        "alphavalue": 0.7,
        "colormap": "rainbow",
        "TrainingFraction": [0.95],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "default",
        "snapshotindex": -1,
        "batch_size": 8,
        "cropping": False,
        "x1": 0, "x2": 640, "y1": 277, "y2": 624,
        "corner2move2": [50, 50],
        "move2corner": True
    }
    return config

def save_config(config, output_folder: Path):
    path = output_folder / "config.yaml"
    yaml_str = ""
    for key, value in config.items():
        if value is None:
            yaml_str += f"{key}:\n"
        else:
            value_str = yaml.dump({key: value}, default_flow_style=False)
            yaml_str += value_str
    path.write_text(yaml_str)
    print(f"✅ Config saved to {path}")

# === SCRIPT TO BUILD LIGHTNING POSE PROJECT ===
#%%
# 1. Set paths
source_root = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_pose_project_2025-06-08")
original_root = Path(r"\\wsl$\Ubuntu\home\sneurobiology\Pose-app\data\dlc10v")
# date_str = datetime.now().strftime("%Y-%m-%d")
# project_root = Path(fr"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_pose_project_{date_str}")
# labeled_data = project_root / "labeled-data"
# videos = project_root / "videos"
# labeled_data.mkdir(parents=True, exist_ok=True)
# videos.mkdir(parents=True)
#%%

lab_data_dir = source_root / "labeled-data"
original_videos_root  = original_root / "videos"
files = [f.name for f in lab_data_dir.iterdir() if f.is_dir()]
original_videos_paths = [Path(f) for f in original_videos_root.glob("*.mp4")]
print(len(original_videos_paths), "original videos found")
#%%
movies_to_move = []
for movie in original_videos_paths:
    print("+1", movie.stem)
    if movie.stem in files:
        movies_to_move.append(movie)
#%%
for movie in movies_to_move:
    print(f"Copying {movie.name} to {lab_data_dir}")
    shutil.copy(movie, source_root / 'videos' / movie.name)

#%%
#extract bodyparts from the first .csv file
def extract_bodyparts_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=[0, 1, 2])
    if isinstance(df.columns, pd.MultiIndex):
        return list(df.columns.levels[1])
    else:
        raise ValueError("CSV does not have MultiIndex format expected by DeepLabCut")
bodyparts = extract_bodyparts_from_csv(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_pose_project_2025-06-08\labeled-data\multicam_video_2024-07-22T10_19_22_cropped_20250325101012\CollectedData.csv")
#%%
bodyparts = bodyparts[3:]
bodyparts

#%%
video_paths = [str(f) for f in (source_root / "videos").glob("*.mp4")]
print(len(video_paths), "video paths found")
print(video_paths[:5])  # Print first 5 video paths for verification
#%%
config = generate_config("movement", source_root, video_paths, bodyparts)
save_config(config, source_root)
#%%
# delete .idenfier files 
import os
import glob


# Recursively search for matching files
pattern = os.path.join(lab_data_dir, '**', '*Zone.Identifier')
matches = glob.glob(pattern, recursive=True)

for file_path in matches:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except FileNotFoundError:
        print(f"File not found (might be on a non-NTFS filesystem): {file_path}")
    except OSError as e:
        print(f"Error deleting {file_path}: {e}")

#%%

# 2. Initialize trackers
video_paths = []
bodyparts = None

# 3. Loop through permutation folders
for folder in sorted(source_root.glob("*permutation")):
    video_file = next(folder.glob("*.mp4"), None)
    h5_file = folder / "CollectedData.h5"
    frames_dir = folder / "extracted_frames"

    if not video_file or not h5_file.exists() or not frames_dir.exists():
        print(f"⚠️ Skipping {folder.name} — missing files")
        continue

    name = video_file.stem
    dst_video_path = videos / video_file.name
    dst_label_dir = labeled_data / name
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    # Copy data
    shutil.copy(video_file, dst_video_path)
    shutil.copy(h5_file, dst_label_dir / "CollectedData.h5")
    csv_file = folder / "CollectedData.csv"
    if csv_file.exists():
        shutil.copy(csv_file, dst_label_dir / "CollectedData.csv")
    else:
        print(f"⚠️ No CollectedData.csv found in {folder.name}")
    for frame in frames_dir.glob("*.png"):
        shutil.copy(frame, dst_label_dir / frame.name)

    video_paths.append(dst_video_path)

    # Extract bodyparts from first .h5
    if bodyparts is None:
        df = pd.read_hdf(h5_file)
        if isinstance(df.columns, pd.MultiIndex):
            bodyparts = list(df.columns.levels[0])
        else:
            raise ValueError("CSV/HDF does not have MultiIndex format expected by DeepLabCut")

    print(f"✅ Added: {folder.name}")

# 4. Create config.yaml
if bodyparts:
    config = generate_config("movement", project_root, video_paths, bodyparts)
    save_config(config, project_root)


