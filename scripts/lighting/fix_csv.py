import os
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

# --- Paths ---
main_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_pose_project_2025-06-08")
labeled_data_dir = main_dir / 'labeled-data'
videos_dir = main_dir / 'videos'

reference_csv_path = Path(
    r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\dlc10v\labeled-data\multicam_video_2024-07-22T10_19_22_cropped_20250325101012\CollectedData.csv"
)

# --- Load reference DLC CSV format ---
ref_df = pd.read_csv(reference_csv_path, header=[0, 1, 2])
pose_columns = ref_df.columns[3:]  # skip first 3 metadata columns
scorer_row = pd.read_csv(reference_csv_path, header=None).iloc[0, 3:]  # scorer row for pose columns only

# --- Process one session ---
for session in tqdm(os.listdir(labeled_data_dir), desc="Processing sessions"):
    session_path = labeled_data_dir / session
    if not session_path.is_dir():
        continue

    collected_data_path = session_path / "CollectedData.csv"
    if not collected_data_path.exists():
        print(f"Skipping {session}: no CollectedData.csv found.")
        continue

    # --- Frame list ---
    frame_files = sorted(
        [f for f in os.listdir(session_path) if re.match(r"frame_\d+\.png", f)],
        key=lambda x: int(re.findall(r"\d+", x)[0])
    )
    if not frame_files:
        print(f"⚠ No frames found in {session}")
        continue

    print(f"{session}: found {len(frame_files)} frames")

    # --- Load broken CSV ---
    broken_df = pd.read_csv(collected_data_path, header=None)

    # --- Ensure correct column count ---
    expected_pose_cols = len(pose_columns)
    if broken_df.shape[1] != expected_pose_cols:
        print(f"🔧 Adjusting columns: found {broken_df.shape[1]}, expected {expected_pose_cols}")
        if broken_df.shape[1] < expected_pose_cols:
            for i in range(expected_pose_cols - broken_df.shape[1]):
                broken_df[f"pad_{i}"] = ""
        else:
            broken_df = broken_df.iloc[:, :expected_pose_cols]

    n_rows = len(frame_files)
    pose_df = pd.DataFrame(broken_df.values[:n_rows], columns=pose_columns)

    # --- Header rows ---
    header_rows = [
        ['scorer', '', ''] + list(scorer_row),
        ['bodyparts', '', ''] + [h[1] for h in pose_columns],
        ['coords', '', ''] + [h[2] for h in pose_columns]
    ]

    # --- Metadata columns: A, B, C
    col_0 = ['labeled-data'] * n_rows
    col_1 = [f"{session}"] * n_rows
    col_2 = frame_files

    meta_df = pd.DataFrame({
        0: col_0,
        1: col_1,
        2: col_2
    })

    # --- Final full table ---
    full_df = pd.concat([meta_df, pose_df.reset_index(drop=True)], axis=1)
    full_df.columns = range(full_df.shape[1])  # numerical columns

    # --- Prepend header rows ---
    header_df = pd.DataFrame(header_rows)
    full_with_header = pd.concat([header_df, full_df], ignore_index=True)

    # --- Save CSV ---
    fixed_csv = session_path / "CollectedData.csv"
    if fixed_csv.exists():
        fixed_csv.unlink()
    full_with_header.to_csv(fixed_csv, header=False, index=False)
    print(f"✅ Fixed CSV saved to: {fixed_csv}")

    # --- Save HDF5 with pose only ---
    pose_hdf = pd.concat([
        pd.DataFrame([scorer_row], columns=pose_columns),
        pose_df.reset_index(drop=True)
    ], ignore_index=True)
    h5_path = session_path / "CollectedData.h5"
    pose_hdf.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
    print(f"💾 HDF5 written: {h5_path}")

