#%%
from pathlib import Path
from movement.io.save_poses import to_dlc_file
import xarray as xr


# get all the directries with triangulated_points.h5 files and save new _dlc.h5 files for each one of them inside the dir


# main_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting"
# # main_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\test_cropping\try_model"
# # save_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\data_newm"
# save_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\dlc_3d_data"
# files = []
# for dir in Path(main_dir).rglob("*"):
#     if dir.is_dir():
# #         for file in dir.rglob("*_triangulated_points.h5"):
#             files.append(file)
# files_set = set(files)

# print(files_set)

# for file in list(files_set):
#     data = xr.open_dataset(file)
#     to_dlc_file(data, Path(save_dir).with_name(Path(file).stem + "_dlc.h5"))
#     print('File saved:', Path(save_dir).with_name(Path(file).stem + "_dlc.h5"))
# print(f"converted {len(files_set)} files")

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

def convert_h5_to_dlc(h5_files: list[Path], save_dir: Path) -> None:
    """
    Convert the given h5 files to DLC format and save them.
    
    Args:
        h5_files (list[Path]): List of h5 files to convert
        save_dir (Path): Directory to save the converted files
    """
    for h5_file in h5_files:
        data = xr.open_dataset(h5_file)
        output_path = Path(save_dir) / f"{h5_file.stem}_dlc.h5"
        to_dlc_file(data, output_path)
        print('File saved:', output_path)
    print(f"Converted {len(h5_files)} files")

# Example usage
if __name__ == "__main__":
    main_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting")
    save_dir = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\dlc_3d_data")
    
    # First find the files
    h5_files = find_dirs_with_matching_views(main_dir)
    print("Found files:", len(h5_files))
    
    # Then convert them if desired
    convert_h5_to_dlc(h5_files, save_dir)
# %%
