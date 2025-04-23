# %%
from movement.io.save_poses import to_dlc_file
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr


# Stack keypoints and view dimensions to create combined labels
def stack_keypoints_views(dataset):
    keypoints = dataset.keypoints.values
    views = dataset.view.values

    # Create new combined labels
    new_keypoint_labels = []
    for view in views:
        for keypoint in keypoints:
            combined_label = f"{keypoint}_{view}"
            new_keypoint_labels.append(combined_label)

    # Stack position and confidence data using a temporary dimension name
    stacked_pos = dataset.position.stack(combined_keypoints=("view", "keypoints"))
    stacked_conf = dataset.confidence.stack(combined_keypoints=("view", "keypoints"))

    # Rename the stacked dimension coordinates
    stacked_pos = stacked_pos.assign_coords(combined_keypoints=new_keypoint_labels)
    stacked_conf = stacked_conf.assign_coords(combined_keypoints=new_keypoint_labels)

    # Create new dataset with stacked dimensions
    stacked_ds = xr.Dataset({"position": stacked_pos, "confidence": stacked_conf})

    # Rename the dimension to 'keypoints'
    stacked_ds = stacked_ds.rename({"combined_keypoints": "keypoints"})

    # Copy attributes from original dataset
    for attr_name, attr_value in dataset.attrs.items():
        stacked_ds.attrs[attr_name] = attr_value

    return stacked_ds


def export_ds_to_dlc_annotation(
    labels_ds: xr.Dataset,
    frames_names: list,
    main_output_folder: Path,
    video_name: str = "test_movie.avi",
):

    video_name = video_name.split(".")[0]
    final_name_csv = f"{video_name}_movement.csv"
    final_name_h5 = f"{video_name}_movement.h5"
    
    main_output_folder = Path(main_output_folder)
    output_folder = main_output_folder / video_name
    output_folder.mkdir(parents=True, exist_ok=True)

    TEMP_LABELS_INDIVIDUAL_FILENAME = "labels_individual_0.h5"
    TEMP_LABELS_FILENAME = "labels.h5"
    to_dlc_file(labels_ds, output_folder / TEMP_LABELS_FILENAME, split_individuals=True)
    df = pd.read_hdf(output_folder / TEMP_LABELS_INDIVIDUAL_FILENAME)

    # change df index to be multiindex with first level "labeled-data", second level movie_name.split(".")[0], third name frames_names:
    df.index = pd.MultiIndex.from_tuples(
        [
            ("labeled-data", movie_name.split(".")[0], frames_names[i])
            for i in range(len(frames_names))
        ]
    )
    df.to_csv(output_folder / final_name_csv) 
    df.to_hdf(output_folder / final_name_h5, key="df", mode="w")



####################
# Integrate those:
# Example movement dataset:

ds = xr.open_dataset(
    "/Volumes/SSD/3D-setup/predictions_multicam_video_2024-07-22T10_19_22_cropped_20250325101012.h5",
    engine="h5netcdf",
)
# Random movie name:
movie_name = "test_movie.avi"
# Random frame idxs:
frame_idxs = np.arange(0, 100)

target_folder = Path("/Users/vigji/Desktop/dummy_project")

####################

# along the keypoints dimension, drop all keypoints that contain "limbmid"
ds = ds.isel(keypoints=~ds.keypoints.str.contains("limbmid"))
ds.position.shape
# Stack entries along keypoint axis, making new labels
ds_stacked = stack_keypoints_views(ds)

# Name of the frames we'll save in the labels folder:
frames_names = [f"frame_{i:05d}.png" for i in frame_idxs]

# create a new folder
target_folder.mkdir(parents=True, exist_ok=True)

labels_folder = target_folder / "labels"
if labels_folder.exists():
    for file in labels_folder.rglob("*"):
        if file.is_file():
            file.unlink()
labels_folder.mkdir(parents=True, exist_ok=True)

# take first 100 values over the time axis of the xarray Dataset:
labels_ds = ds_stacked.isel(time=frame_idxs)

export_ds_to_dlc_annotation(labels_ds, frames_names, labels_folder, movie_name)
# %%
import flammkuchen as fl

test_h5 = '/Users/vigji/Desktop/mouseface-Luigi Petrucco-2022-06-15/labeled-data/Basler_acA1440-220um__40075240__20220608_122024069/CollectedData_Luigi Petrucco.h5'
fl_ds = fl.load(test_h5)
correct_df = pd.read_hdf(test_h5)
to_test_df = pd.read_hdf("/Users/vigji/Desktop/dummy_project/labels/test_movie/test_movie_movement.h5")
# %%
to_test_df.head()
# %%
correct_df.head()
# %%
to_test_df.iloc[:, 1:]
# %%
correct_df.columns
# %%
to_test_df.columns
# %%
