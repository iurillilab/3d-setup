# %%
import napari
import numpy as np
from movement.napari.layer_styles import TracksStyle, PointsStyle
from movement.napari.utils import columns_to_categorical_codes
from movement.napari.convert import ds_to_napari_tracks, _replace_nans_with_zeros
from movement.filtering import filter_by_confidence, interpolate_over_time

from movement import sample_data

file_names = sample_data.list_datasets()
print(*file_names, sep='\n')  # print each sample file in a separate line
# ?sample_data.fetch_dataset
# %%
ds_name = "SLEAP_single-mouse_EPM.predictions.slp"
ds = sample_data.fetch_dataset(ds_name)
ds_filtered = filter_by_confidence(ds, threshold=0.6, print_report=True)
ds_interpolated = interpolate_over_time(
    ds_filtered, method="linear", max_gap=1, print_report=True
)

# print(ds["position"])
# self.data, self.props = ds_to_napari_tracks(ds)
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


viewer = napari.Viewer()
# Add the new layers to the napari viewer
viewer.add_tracks(track) # , **tracks_style.as_kwargs())
viewer.add_points(track[:, 1:]) # , **points_style.as_kwargs())
# %%
