from movement.io.save_poses import to_dlc_file
from movement.io.load_poses import from_numpy
from movement.io.load_poses import from_file
from pathlib import Path
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse


def find_dirs_with_matching_views(root_dir: Path) -> list[Path]:
    """
    Find directories containing exactly 5 SLP files with matching camera views.
    """
    valid_dirs = []

    all_candidate_folders = [f for f in root_dir.rglob("multicam_video_*_cropped_*") if f.is_dir()]
    parent_dict = {folder.parent: [] for folder in all_candidate_folders}
    
    for candidate_folder in all_candidate_folders:
        parent_dict[candidate_folder.parent].append(candidate_folder)
    
    last_folders = [sorted(folders)[-1] for folders in parent_dict.values()]


    for directory in last_folders:    
        #if not directory.is_dir():
        #    continue

        if "calibration" in [parent.name.lower() for parent in directory.parents]:
            continue
        # Get all SLP files in the current directory
        slp_files = list(directory.glob('*.slp'))

        if  len(list(directory.glob("*triangulated_points_*.h5"))) < 0:
            continue
        
        for h5 in directory.glob("*triangulated_points_*.h5"):
            if not h5.is_file():
                continue
            valid_dirs.append(h5)
    valid_dirs.reverse() # to avoid possible error 
    return valid_dirs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Find directories with matching camera views")
    parser.add_argument("root_dir", type=str, help="Root directory to search for SLP files")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    valid_dirs = find_dirs_with_matching_views(root_dir)

    for dir in valid_dirs:
        print(dir)

