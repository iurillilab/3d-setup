import pandas as pd
from pathlib import Path
import os
import re

# --- Define paths ---
reference_csv = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\dlc10v\labeled-data\multicam_video_2024-07-22T10_19_22_cropped_20250325101012\CollectedData.csv")          # ✅ GOOD file
broken_csv = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_pose_project_2025-06-08\labeled-data\multicam_video_2024-07-22T10_19_22_cropped_20250325101012\CollectedData.csv")           # ❌ BROKEN file
frame_dir = broken_csv.parent
video_name = broken_csv.parent.name                          # used as the video label (no .mp4)

# --- Load good reference structure ---
ref_df = pd.read_csv(reference_csv, header=[0, 1, 2])
pose_columns = ref_df.columns[3:]
scorer_row = ref_df.iloc[0, 3:]

# --- Get frame list ---
frame_files = sorted(
    [f for f in os.listdir(frame_dir) if re.match(r"frame_\d+\.png", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

# --- Load broken pose data (no headers) ---
broken_df = pd.read_csv(broken_csv, header=None)

# --- Align row count to frames ---
n_frames = len(frame_files)
if broken_df.shape[0] < n_frames:
    pad = pd.DataFrame([[None]*broken_df.shape[1]] * (n_frames - broken_df.shape[0]))
    broken_df = pd.concat([broken_df, pad], ignore_index=True)
elif broken_df.shape[0] > n_frames:
    broken_df = broken_df.iloc[:n_frames]

# --- Align column count to pose columns ---
if broken_df.shape[1] < len(pose_columns):
    for i in range(len(pose_columns) - broken_df.shape[1]):
        broken_df[f"pad_{i}"] = None
elif broken_df.shape[1] > len(pose_columns):
    broken_df = broken_df.iloc[:, :len(pose_columns)]

pose_df = pd.DataFrame(broken_df.values, columns=pose_columns)

# --- Metadata columns ---
meta_0 = ['scorer', 'bodyparts', 'coords'] + ['labeled-data'] * n_frames
meta_1 = ['', '', ''] + [video_name] * n_frames
meta_2 = ['', '', ''] + frame_files

# --- Assemble full DataFrame ---
full_df = pd.DataFrame({
    0: meta_0,
    1: meta_1,
    2: meta_2
})
for i, col in enumerate(pose_columns):
    full_df[i + 3] = [scorer_row[i], col[1], col[2]] + pose_df.iloc[:, i].tolist()

# --- Save as clean DLC-compatible file ---
fixed_csv_path = broken_csv.parent / "CollectedData_FIXED.csv"
full_df.to_csv(fixed_csv_path, header=False, index=False)
print(f"✅ Fixed CSV written to: {fixed_csv_path}")
