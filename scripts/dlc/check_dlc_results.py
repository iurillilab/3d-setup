# %%
from pathlib import Path
import pandas as pd
import time
from functools import lru_cache
import re
from typing import Dict, Optional, Tuple, List
from pprint import pprint
from tqdm import tqdm
from collections import defaultdict
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
        "session": get_session_from_path(fpath),
    }
    parsed_info.update(updates_dict)
    return parsed_info

all_dlc_files_info = [get_all_info_from_fpath(fpath) for fpath in tqdm(all_h5_paths)]
all_dlc_files_df = pd.DataFrame(all_dlc_files_info)
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
all_cropped_files_df["original_video_fpath"].unique()
# %%

def generate_tracking_report(all_cropped_files_df: pd.DataFrame, all_dlc_files_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a report checking if each original video has adequate tracking coverage.
    
    For each original video, check:
    1. If there's a single crop with both session entity and mouse tracking
    2. If not, whether combining multiple crops can provide both models
    3. Which entities are missing for complete tracking
    4. Whether the video has at least one loadable crop
    
    Returns:
        DataFrame with columns:
        - original_video_fpath: path to original video
        - has_valid_single_crop: bool, if single crop has both models
        - valid_single_crop_name: crop name if has_valid_single_crop is True
        - can_pool_crops: bool or None, if pooling multiple crops works (None if single crop works)
        - pooling_crops: tuple of crop names or None
        - missing_entities: set of entities that need to be tracked to complete the analysis
        - has_loadable_crop: bool, whether video has at least one loadable crop
    """
    
    # Get unique original videos
    original_videos = all_cropped_files_df['original_video_fpath'].unique()
    
    report_data = []
    
    for original_video in original_videos:
        # Get session name for this original video
        video_crops = all_cropped_files_df[all_cropped_files_df['original_video_fpath'] == original_video]
        if len(video_crops) == 0:
            continue
            
        session_name = video_crops['session'].iloc[0]  # Assuming consistent session per original video
        
        # Get mouse ID and day from the cropped files dataframe
        mouse_id = video_crops['mouse'].iloc[0] if 'mouse' in video_crops.columns else None
        day = video_crops['day'].iloc[0] if 'day' in video_crops.columns else None
        
        # Check if video has at least one loadable crop
        has_loadable_crop = False
        if 'loadable' in video_crops.columns:
            has_loadable_crop = video_crops['loadable'].any()
        else:
            # If loadable column doesn't exist, assume all crops are loadable
            has_loadable_crop = len(video_crops) > 0
        
        # Assert that the video has at least one loadable crop
        # assert has_loadable_crop, f"Original video {original_video} has no loadable crops"
        
        # Get all DLC results for this original video
        dlc_results = all_dlc_files_df[all_dlc_files_df['original_video_fpath'] == original_video]
        
        # Group DLC results by crop
        crops_with_tracking = defaultdict(set)
        for _, dlc_row in dlc_results.iterrows():
            crop_path = dlc_row['cropped_video_fpath']
            entity = dlc_row['entity']
            crops_with_tracking[crop_path].add(entity)
        
        # Check for single crop solution
        valid_single_crop = None
        required_entities = {session_name, 'mouse'}
        
        # Get all available entities across all crops
        all_available_entities = set()
        for entities in crops_with_tracking.values():
            all_available_entities.update(entities)
        
        # Calculate missing entities
        missing_entities = required_entities - all_available_entities
        
        for crop_path, entities in crops_with_tracking.items():
            if required_entities.issubset(entities):
                valid_single_crop = crop_path
                break
        
        # If single crop works
        if valid_single_crop is not None:
            report_data.append({
                'original_video_fpath': original_video,
                'has_valid_single_crop': True,
                'valid_single_crop_name': valid_single_crop.name if hasattr(valid_single_crop, 'name') else str(valid_single_crop),
                'can_pool_crops': None,
                'pooling_crops': None,
                'missing_entities': set(),  # No missing entities if single crop works
                'has_loadable_crop': has_loadable_crop,
                'mouse': mouse_id,
                'day': day
            })
            continue
        
        # Check if pooling multiple crops can work
        crops_with_entities = []
        
        for crop_path, entities in crops_with_tracking.items():
            if entities:  # Only include crops that have some tracking
                crops_with_entities.append((crop_path, entities))
        
        # Check if we can get both required entities from multiple crops
        can_pool = required_entities.issubset(all_available_entities)
        pooling_crops = None
        
        if can_pool:
            # Find minimal set of crops that covers both entities
            session_crops = [crop for crop, entities in crops_with_entities if session_name in entities]
            mouse_crops = [crop for crop, entities in crops_with_entities if 'mouse' in entities]
            
            if session_crops and mouse_crops:
                # Use the first crop that has session entity and first that has mouse
                # (could be optimized to find overlap or minimal set)
                pooling_set = set()
                pooling_set.add(session_crops[0])
                pooling_set.add(mouse_crops[0])
                pooling_crops = tuple(crop.name if hasattr(crop, 'name') else str(crop) for crop in pooling_set)
        
        report_data.append({
            'original_video_fpath': original_video,
            'has_valid_single_crop': False,
            'valid_single_crop_name': None,
            'can_pool_crops': can_pool,
            'pooling_crops': pooling_crops,
            'missing_entities': missing_entities,
            'has_loadable_crop': has_loadable_crop,
            'mouse': mouse_id,
            'day': day
        })
    
    return pd.DataFrame(report_data)

# Generate the report
tracking_report = generate_tracking_report(all_cropped_files_df, all_dlc_files_df)
print("Tracking Report Summary:")
print(f"Total original videos: {len(tracking_report)}")
print(f"Videos with valid single crop: {tracking_report['has_valid_single_crop'].sum()}")
print(f"Videos that can use pooling: {tracking_report['can_pool_crops'].sum()}")
print(f"Videos with no solution: {len(tracking_report) - tracking_report['has_valid_single_crop'].sum() - tracking_report['can_pool_crops'].sum()}")
print(f"Videos with loadable crops: {tracking_report['has_loadable_crop'].sum()}")
print(f"Videos missing entities: {(tracking_report['missing_entities'].apply(len) > 0).sum()}")

tracking_report

# %%
# First, analyze tracking coverage for original videos
print("="*60)
print("TRACKING COVERAGE ANALYSIS")
print("="*60)

# Find all possible mouse x day combinations from cropped files that have tracking
tracking_combinations = set(zip(tracking_report["mouse"], tracking_report["day"]))
cropped_combinations = set(zip(all_cropped_files_df["mouse"], all_cropped_files_df["day"]))

# Find combinations that have cropped videos but no tracking data
missing_tracking_combinations = cropped_combinations - tracking_combinations
print(f"Mouse x day combinations with cropped videos: {len(cropped_combinations)}")
print(f"Mouse x day combinations with tracking data: {len(tracking_combinations)}")
print(f"Mouse x day combinations missing tracking: {len(missing_tracking_combinations)}")

if missing_tracking_combinations:
    print(f"\nCombinations with cropped videos but no tracking data:")
    for mouse, day in sorted(missing_tracking_combinations):
        # Get sessions available for this combination
        combo_sessions = all_cropped_files_df[
            (all_cropped_files_df["mouse"] == mouse) & 
            (all_cropped_files_df["day"] == day)
        ]["session"].unique()
        print(f"  {mouse} - {day}: sessions {list(combo_sessions)}")

# Show detailed breakdown of tracking issues
incomplete_tracking = tracking_report[~tracking_report["has_valid_single_crop"]]
if len(incomplete_tracking) > 0:
    print(f"\nOriginal videos with incomplete tracking ({len(incomplete_tracking)}):")
    for _, row in incomplete_tracking.iterrows():
        missing_entities_str = ', '.join(row['missing_entities']) if row['missing_entities'] else 'none'
        can_pool = "Yes" if row['can_pool_crops'] else "No"
        print(f"  {row['mouse']} - {row['day']}: missing entities [{missing_entities_str}], can pool: {can_pool}, has loadable crop: {row['has_loadable_crop']}")

print(f"\n")
print("="*60)
print("SESSION COVERAGE ANALYSIS")
print("="*60)

# Check session coverage for each mouse x day combination
# Each combination should have:
# 1. At least 1 "object" session
# 2. At least 1 "cricket" or "roach" session

# Group by mouse x day and get available sessions
session_coverage = all_cropped_files_df.groupby(['mouse', 'day'])['session'].apply(set).reset_index()
session_coverage.columns = ['mouse', 'day', 'available_sessions']

print(f"Total mouse x day combinations: {len(session_coverage)}")

# Check each combination for required sessions
incomplete_combinations = []
missing_object = []
missing_animal = []

for _, row in session_coverage.iterrows():
    mouse = row['mouse']
    day = row['day']
    sessions = row['available_sessions']
    
    has_object = 'object' in sessions
    has_animal = any(animal in sessions for animal in ['cricket', 'roach'])
    
    if not has_object:
        missing_object.append((mouse, day, sessions))
    
    if not has_animal:
        missing_animal.append((mouse, day, sessions))
    
    if not (has_object and has_animal):
        incomplete_combinations.append((mouse, day, sessions))

print(f"\nSession Coverage Analysis:")
print(f"Complete combinations (object + cricket/roach): {len(session_coverage) - len(incomplete_combinations)}")
print(f"Incomplete combinations: {len(incomplete_combinations)}")
print(f"Missing 'object' session: {len(missing_object)}")
print(f"Missing 'cricket'/'roach' session: {len(missing_animal)}")

if missing_object:
    print(f"\nCombinations missing 'object' session:")
    for mouse, day, sessions in sorted(missing_object):
        print(f"  {mouse} - {day}: has {sessions}")

if missing_animal:
    print(f"\nCombinations missing 'cricket'/'roach' session:")
    for mouse, day, sessions in sorted(missing_animal):
        print(f"  {mouse} - {day}: has {sessions}")

if incomplete_combinations:
    print(f"\nAll incomplete combinations:")
    for mouse, day, sessions in sorted(incomplete_combinations):
        missing = []
        if 'object' not in sessions:
            missing.append('object')
        if not any(animal in sessions for animal in ['cricket', 'roach']):
            missing.append('cricket/roach')
        print(f"  {mouse} - {day}: has {sessions}, missing {missing}")
else:
    print("\nAll mouse x day combinations have complete session coverage!")

# %%
tracking_report[~tracking_report["has_valid_single_crop"]]
# %%
all_cropped_files_df.columns
# M31: has 3 sessions for 20250510
# M30: has 3 sessions for day 13
# M30: has one missing session, object, for day 9
# %%
