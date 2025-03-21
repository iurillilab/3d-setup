#%%
from pathlib import Path
from movement.io.save_poses import to_dlc_file
import xarray as xr


# get all the directries with triangulated_points.h5 files and save new _dlc.h5 files for each one of them inside the dir


#main_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting"
main_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\test_cropping\try_model"
save_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\data_newm"
#ave_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\dlc_3d_data"
files = []
for dir in Path(main_dir).rglob("*"):
    if dir.is_dir():
        for file in dir.rglob("*_triangulated_points.h5"):
            files.append(file)
files_set = set(files)

print(files_set)

for file in list(files_set):
    data = xr.open_dataset(file)
    to_dlc_file(data, Path(save_dir).with_name(Path(file).stem + "_dlc.h5"))
    print('File saved:', Path(save_dir).with_name(Path(file).stem + "_dlc.h5"))
print(f"converted {len(files_set)} files")

#%%