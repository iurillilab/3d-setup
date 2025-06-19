# %%
from pathlib import Path
import pandas as pd
import time
from functools import lru_cache
import re
from typing import Dict, Optional
from pprint import pprint
from tqdm import tqdm
# %%
@lru_cache(maxsize=10000)
def get_all_h5_paths(root_path):
    return [f for f in root_path.rglob("*snapshot*.h5") if "test-output" not in str(f)]

root_path = Path("/Volumes/SNeurobiology_RAW/nas_mirror")
all_h5_paths = get_all_h5_paths(root_path)

# %%

def parse_dlc_filename(filepath: Path) -> Optional[Dict[str, str]]:
    """
    Parse DLC model information from h5 filename.
    
    Args:
        filepath: Path to the h5 file
        
    Returns:
        Dictionary with video_filename, model_name, architecture, shuffle, and snapshot
        or None if parsing fails
    """
    filename = filepath.name
    
    # Pattern: {video_filename}DLC_{architecture}_{model_name}shuffle{shuffle}_snapshot_{snapshot}.h5
    # Example: multicam_video_2025-05-07T14_11_04_central.aviDLC_HrnetW48_dlc3_cricketJun10shuffle8_snapshot_177.h5
    pattern = r'(.+?)DLC_([^_]+)_(.+?)shuffle(\d+)_snapshot_(\d+)\.h5$'
    
    match = re.search(pattern, filename)
    if match:
        video_filename, architecture, model_name, shuffle, snapshot = match.groups()
        return {
            'cropped_video_filestem': video_filename,
            'architecture': architecture,
            'model_name': model_name,
            'shuffle': shuffle,
            'snapshot': snapshot
        }
    return None

def get_dlc_h5_len(h5_path: Path) -> int:
    """Get length of DLC HDF5 file efficiently without reading data into memory."""
    with pd.HDFStore(str(h5_path), mode='r') as store:
        key = list(store.keys())[0]
        return store.get_storer(key).nrows
    
def get_entity_from_modelname(modelname: str) -> str:
    possible_entities = ["cricket", "mouse", "object", "roach"]
    for entity in possible_entities:
        if entity in modelname:
            return entity
    raise ValueError(f"No entity found in modelname: {modelname}")

# Test the parsing function with your example
fpath = Path('/Volumes/SNeurobiology_RAW/nas_mirror/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped_20250528154623/multicam_video_2025-05-07T14_11_04_central.aviDLC_HrnetW48_dlc3_cricketJun10shuffle8_snapshot_177.h5')


def get_all_info_from_fpath(fpath: Path):
    parsed_info = parse_dlc_filename(fpath)

    cropped_video_fpaths = list(fpath.parent.glob(f"{parsed_info['cropped_video_filestem']}.mp4"))
    original_video_fpaths = list(fpath.parent.parent.glob(f"{parsed_info['cropped_video_filestem'][:34]}*.avi"))

    assert len(cropped_video_fpaths) < 2, f"Multiple cropped video files found for {parsed_info['cropped_video_filestem']}"
    assert len(original_video_fpaths) < 2, f"Multiple original video files found for {parsed_info['cropped_video_filestem']}"

    updates_dict = {
        "file_length": get_dlc_h5_len(fpath),
        "cropped_video_fpath": cropped_video_fpaths[0] if cropped_video_fpaths else None,
        "original_video_fpath": original_video_fpaths[0] if original_video_fpaths else None,
        "original_video_filestem": original_video_fpaths[0].stem if original_video_fpaths else None,
        "has_cropped": len(cropped_video_fpaths),
        "has_original": len(original_video_fpaths),
        "entity": get_entity_from_modelname(parsed_info["model_name"]),
        "session": get_entity_from_modelname(str(fpath.parent)),
    }
    parsed_info.update(updates_dict)
    return parsed_info

all_dlc_files_df = [get_all_info_from_fpath(fpath) for fpath in tqdm(all_h5_paths)]
all_dlc_files_df = pd.DataFrame(all_info)
all_dlc_files_df.head()
# %%
# "full_model_id": f"{parsed_info['model_name']}_{parsed_info['shuffle']}_{parsed_info['snapshot']}",

all_dlc_files_df["full_model_id"] = all_dlc_files_df.apply(lambda x: f"{x['model_name']}_{x['shuffle']}_{x['snapshot']}", axis=1)
all_dlc_files_df.full_model_id.unique()
# %%
all_cropped_files_df = pd.read_csv("/Users/vigji/code/3d-setup/video_integrity_report.csv")
all_cropped_files_df["original_video_fpath"] = all_cropped_files_df["original"].apply(lambda x: Path(x))
all_cropped_files_df["cropped_video_fpath"] = all_cropped_files_df["cropped_file"].apply(lambda x: Path(x))
# %%
assert all_dlc_files_df["original_video_fpath"].isin(all_cropped_files_df["original_video_fpath"]).all()

# %%
all_dlc_files_df["original_video_fpath"][0]
# %%
all_cropped_files_df["original_video_fpath"][0]
# %%
