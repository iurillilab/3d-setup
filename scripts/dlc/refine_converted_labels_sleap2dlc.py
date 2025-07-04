import pandas as pd
from pathlib import Path
import shutil
from pprint import pprint
import yaml
from typing import Dict, Any
import cv2
from copy import deepcopy

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file as dictionary."""
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save dictionary as YAML config file."""
    with config_path.open('w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

converted_from_sleap_dir = "/Users/vigji/Desktop/almost_final_models/dlc3_roach-YaduLuigi-2025-06-10"
target_folder = "/Users/vigji/Desktop/roach-bottom-vigji-2025-07-01"

FROM_LABEL_FORMAT = "h5"   # or "pkl"
DEFAULT_EXTENSION = ".mp4"
NEW_SCORER_NAME = "vigji"
# side labels:
# conversion_dict = {"nose": "nose",
#                     "ear_lf": "Lear",
#                     "forepaw_lf": "LFlimb",
#                     "hindpaw_lf": "LBlimb",
#                     "tailbase": "tailbase",
#                     "hindpaw_rt": "RBlimb",
#                     "forepaw_rt": "RFlimb",
#                     "ear_rt": "Rear",
#                     "belly_rostral": "FLimbMid",
#                     "belly_caudal": "BLimbMid",
#                     "back_caudal": "UpperBack",
#                     "back_mid": "UpperMid",
#                     "back_rostral": "UpperForward"}
# bottom labels:
# conversion_dict = {"nose": "nose",
#                     "ear_lf": "Lear",
#                     "forepaw_lf": "LFlimb",
#                     "hindpaw_lf": "LBlimb",
#                     "tailbase": "tailbase",
#                     "hindpaw_rt": "RBlimb",
#                     "forepaw_rt": "RFlimb",
#                     "ear_rt": "Rear",
#                     "belly_rostral": "Flimbmid",
#                     "belly_caudal": "Blimbmid",
#                     "back_caudal": "UpperBack",
#                     "back_mid": "UpperMid",
#                     "back_rostral": "UpperForward"}
# conversion_dict = {"head": "cricket_front",
#                    "abdomen": "cricket_rear"}
    # conversion_dict = {"tip": "object_front",
    #                 "lid": "object_rear"}
conversion_dict = {"head": "roach_front",
                   "abdomen": "roach_rear",
                   "back": "roach_back"}


converted_from_sleap_dir = Path(converted_from_sleap_dir)

config_dict = load_config(converted_from_sleap_dir / "config.yaml")
pprint(config_dict)

labels_main_dir = converted_from_sleap_dir / "labeled-data"
all_labels_folders = [f for f in (labels_main_dir.glob("*")) if f.is_dir()]
all_labels_folders = sorted(all_labels_folders)
pprint(all_labels_folders)

target_folder = Path(target_folder)

target_config = load_config(target_folder / "config.yaml")
target_bdyparts_order = tuple(target_config["bodyparts"])
target_labels_dir = target_folder / "labeled-data"

target_config_updated = deepcopy(target_config)
if target_config_updated["video_sets"] is None:
    target_config_updated["video_sets"] = {}

# Create inverse conversion dict (DLC -> SLEAP)
inverse_conversion_dict = {v: k for k, v in conversion_dict.items()}


source_labels = [f for f in labels_main_dir.glob(f"*/*.{FROM_LABEL_FORMAT}")]

for source_label in source_labels:
    print(source_label)
    if FROM_LABEL_FORMAT == "h5":
        df = pd.read_hdf(source_label)
    else:
        df = pd.read_pickle(source_label)
    print(df.columns)
    # Get the current MultiIndex levels
    levels = df.columns.levels
    codes = df.columns.codes

    print(levels[1])
    
    # Remap the second level (body parts) using inverse conversion dict
    new_bodyparts = []
    for bodypart in levels[1]:
        if bodypart in inverse_conversion_dict:
            new_bodyparts.append(inverse_conversion_dict[bodypart])
        else:
            print(f"Warning: {bodypart} not found in conversion dict, excluding")
            # new_bodyparts.append(bodypart)
    
    # Create new MultiIndex with remapped body parts
    new_levels = [levels[0], new_bodyparts, levels[2]]
    df.columns = pd.MultiIndex.from_arrays([
        # df.columns.get_level_values(0),
        [NEW_SCORER_NAME, ] * len(df.columns),
        [inverse_conversion_dict.get(bp, bp) for bp in df.columns.get_level_values(1)],
        df.columns.get_level_values(2)
    ], names=df.columns.names)
    
    
    # Resort columns to match target_bdyparts_order
    # Create new column order
    scorer = df.columns.get_level_values(0)[0]  # Assuming same scorer for all
    new_columns = []
    
    for bodypart in target_bdyparts_order:
        for coord in ['x', 'y']:
            new_columns.append((scorer, bodypart, coord))
    
    # Filter to only include columns that exist in the dataframe
    existing_columns = [col for col in new_columns if col in df.columns]
    
    # Reorder the dataframe
    df_reordered = df[existing_columns]
    
    # Get the current index levels
    level_0 = df_reordered.index.get_level_values(0)
    level_1 = df_reordered.index.get_level_values(1)
    level_2 = df_reordered.index.get_level_values(2)
    
    # Remove .avi from second level
    level_1_cleaned = [str(val).replace(".avi", "") for val in level_1]

    # Create new MultiIndex with cleaned second level
    df_reordered.index = pd.MultiIndex.from_arrays([level_0, level_1_cleaned, level_2], 
                                                    names=df_reordered.index.names)
        
    # Save the processed dataframe:
    # Fix old bug:
    target_folder_name = source_label.parent.name
    target_folder_name = target_folder_name.replace(".avi", "")

    output_folder = target_labels_dir / target_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path_pkl = output_folder / f"CollectedData_{NEW_SCORER_NAME}.pkl"
    index_tuples = df_reordered.index.tolist()

    # reorder dataframe alphabetically based on multiindex:
    df_reordered = df_reordered.sort_index()

    df_reordered.to_pickle(output_path_pkl)
    df_reordered.to_csv(output_path_pkl.with_suffix(".csv"))
    df_reordered.to_hdf(output_path_pkl.with_suffix(".h5"), key="data")
    
    # copy all frames in the folder:
    for frame_file in source_label.parent.glob("*.png"):
        shutil.copy(frame_file, output_folder / frame_file.name)

    # read one frame to get the size:
    frame = cv2.imread(str(frame_file))
    height, width, _ = frame.shape

    new_video_set_dict = {(target_folder_name + DEFAULT_EXTENSION): {"crop": f"0, {width}, 0, {height}"}}
    target_config_updated["video_sets"].update(new_video_set_dict)

save_config(target_config_updated, target_folder / "config.yaml")
save_config(target_config, target_folder / "config_before_merging.yaml")







