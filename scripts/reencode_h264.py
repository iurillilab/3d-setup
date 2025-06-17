import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import glob

root = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_pose_project_2025-06-08\labeled-data")  
extensions = {".mp4", ".mov", ".avi", ".mkv"}

def is_h264(video):
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video)
        ], capture_output=True, text=True)
        return result.stdout.strip() == "h264"
    except Exception as e:
        print(f"ffprobe error on {video}: {e}")
        return False

def reencode(video):
    print(f"Re-encoding {video}...")
    temp = video.with_suffix(".temp.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy", str(temp)
    ])
    os.remove(video)
    temp.rename(video)
def remove_old_cleaned_files(root_dir):
    print("🧹 Removing old *_cleaned.csv and *_cleaned.h5 files...")
    removed = 0
    for ext in ("*.csv", "*.h5"):
        for filepath in glob.glob(os.path.join(root_dir, "**", f"*cleaned.{ext.split('.')[-1]}"), recursive=True):
            try:
                os.remove(filepath)
                removed += 1
            except Exception as e:
                print(f"Failed to delete {filepath}: {e}")
    print(f"✔ Removed {removed} cleaned files\n")

def process_all_csvs(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "CollectedData.csv":
                csv_path = os.path.join(subdir, file)
                h5_path = os.path.join(subdir, "CollectedData.h5")

                print(f"Processing: {csv_path}")

                try:
                    # Try DeepLabCut multi-index format
                    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
                    is_multiindex = isinstance(df.columns, pd.MultiIndex)
                except Exception as e:
                    print(f"  Not multi-index: {e}")
                    try:
                        df = pd.read_csv(csv_path, index_col=0)
                        is_multiindex = False
                    except Exception as e2:
                        print(f"  Skipping {csv_path}: {e2}")
                        continue

                # Remove 'Unnamed' columns
                if is_multiindex:
                    df = df.loc[:, ~df.columns.to_frame().apply(lambda col: col.str.contains('Unnamed').any(), axis=1)]
                else:
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                df.reset_index(drop=True, inplace=True)

                # Overwrite the original CSV
                df.to_csv(csv_path, index=False)

                # Overwrite HDF5
                if os.path.exists(h5_path):
                    os.remove(h5_path)
                df.to_hdf(h5_path, key='df', mode='w')

                print(f"✔ Cleaned and saved: {csv_path} and {h5_path}")
# for file in root.rglob("*"):
#     if file.suffix.lower() in extensions:
#         if not is_h264(file):
#             reencode(file)
#         else:
#             print(f"Skipping {file} (already H.264)")
remove_old_cleaned_files(root)
process_all_csvs(root)


#TODO: add col again, fiish csv processing 