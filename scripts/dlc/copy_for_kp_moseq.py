# %%
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pandas as pd

root_path = Path("/Volumes/SNeurobiology_RAW/nas_mirror")
valid_crop_paths_list = pd.read_csv(root_path / "valid_crops_with_full_tracking.txt", header=None)[0].tolist()
# Copy valid videos & their tracking h5 files for mouse to new dir for kp-moseq

import shutil
target_dest = Path("/Volumes/SNeurobiology_RAW/luigi/sel_for_kp_moseq")
target_dest.mkdir(parents=True, exist_ok=True)
dry_run = False

mapping_dict = {
                "Nose": "nose",
                "Lear": "lf_ear",
                "Rear": "lf_forelimb",
                "Lflimb": "lf_hindlimb",
                "Rflimb": "tailbase",
                "Flimbmid": "rt_hindlimb",
                "Lblimb": "rt_forelimb",
                "Rblimb": "rt_ear",
                "Blimbmid": "forelimb_mid",
                "Tailbase": "hindlimb_mid",
            }

for video_path in tqdm(valid_crop_paths_list):
    video_path = Path(video_path)
    target_video_path = target_dest / video_path.name
    mouse_file = next(video_path.parent.glob(f"*{video_path.stem}*mouse*.h5"))
    
    target_mouse_path = target_dest / mouse_file.name
    if not dry_run:
        shutil.copy(video_path, target_video_path)
        df = pd.read_hdf(mouse_file)

        # Fix columns as needed:
        df = df.rename(columns=mapping_dict, level=1)
        df.to_hdf(target_mouse_path, key="df", mode="w")
        # shutil.copy(mouse_file, target_mouse_path)
    else:
        print(f"Would copy {video_path} to {target_video_path}")
        print(f"Would copy {mouse_file} to {target_mouse_path}")

# %%
print(len(valid_crop_paths_list))
# %%
len(tracking_report[tracking_report['has_valid_single_crop']])

# %%

df = pd.read_hdf(mouse_file)
# %%

df.columns.levels[1]

# 

fixed_df = df.copy()

# Rename the second level of the MultiIndex columns using the mapping dictionary
fixed_df = fixed_df.rename(columns=mapping_dict, level=1)

# Alternative approach if the above doesn't work:
# new_columns = df.columns.to_list()
# for i, (level0, level1) in enumerate(new_columns):
#     if level1 in mapping_dict:
#         new_columns[i] = (level0, mapping_dict[level1])
# fixed_df.columns = pd.MultiIndex.from_tuples(new_columns)
# %%
df.head()
# %%
fixed_df.head()
# %%
