#%%
from pathlib import Path
from movement.io.save_poses import to_dlc_file
import xarray as xr
#%%
file = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240722\M1\101552\multicam_video_2024-07-22T10_19_22_cropped_20241209165236\multicam_video_2024-07-22T10_19_22_cropped_20241209165236_triangulated_points.h5"

data = xr.open_dataset(file)
#%%

to_dlc_file(data, Path(file).with_name(Path(file).stem + "_dlc.h5"))

#%%

# get all the directries with triangulated_points.h5 files and save new _dlc.h5 files for each one of them inside the dir


main_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting"
save_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\dlc_3d_data"
files = []
for dir in Path(main_dir).rglob("*"):
    if dir.is_dir():
        for file in dir.rglob("*_triangulated_points.h5"):
            files.append(file)
files_set = set(files)




for file in list(files_set):
    data = xr.open_dataset(file)
    to_dlc_file(data, Path(save_dir).with_name(Path(file).stem + "_dlc.h5"))
print(f"converted {len(files_set)} files")

#%%