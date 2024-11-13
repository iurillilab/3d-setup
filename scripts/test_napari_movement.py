# %%
import napari
import numpy as np
from movement.napari.layer_styles import TracksStyle, PointsStyle
from movement.napari.utils import columns_to_categorical_codes
from movement.napari.convert import ds_to_napari_tracks

from movement import sample_data

file_names = sample_data.list_datasets()
print(*file_names, sep='\n')  # print each sample file in a separate line
# ?sample_data.fetch_dataset
# %%
ds_name = "SLEAP_single-mouse_EPM.predictions.slp"
ds = sample_data.fetch_dataset(ds_name)
# print(ds["position"])
track, props = ds_to_napari_tracks(ds)

# %%
viewer = napari.Viewer()
points_style = PointsStyle(
    name=f"Keypoints - {ds_name}",
    properties=props,
)
points_style.set_color_by(prop="keypoint", cmap="turbo")

# # Track properties must be numeric, so convert str to categorical codes
tracks_props = columns_to_categorical_codes(
    props, ["individual", "keypoint"]
)

# kwargs for the napari Tracks layer
# Track properties must be numeric, so convert str to categorical codes
tracks_props = columns_to_categorical_codes(
    props, ["individual", "keypoint"]
)
tracks_style = TracksStyle(
    name=f"Tracks - {ds_name}",
    properties=tracks_props,
)
tracks_style.set_color_by(prop="keypoint", cmap="turbo")

# %%
viewer = napari.Viewer()
# Add the new layers to the napari viewer
viewer.add_tracks(ds.data) # , **tracks_style.as_kwargs())
viewer.add_points(ds.data[:, 1:]) # , **points_style.as_kwargs())
# %%
