# %%
%matplotlib widget
from pathlib import Path
import napari
from threed_utils.detection_napari_check import view_movement_3d
from threed_utils.io import load_triangulated_ds, sanitize_keypoints
from movement.io.load_poses import from_file
from matplotlib import pyplot as plt
import numpy as np


data_path = Path("/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021_triangulated_points_20250730-215649.h5")
bottom_view_ds = from_file(next(data_path.parent.glob("*mouse-bottom*.h5")), source_software="DeepLabCut")
triang_ds = load_triangulated_ds(data_path)

bottom_view_ds = bottom_view_ds.sel(time=slice(0, triang_ds.time.isel(time=-1)))
bottom_view_ds = sanitize_keypoints(bottom_view_ds)

# Process x and y coordinates separately: subtract min and scale to match triangulated data
for space_coord in ['x', 'y']:
    # Subtract min from bottom_view_ds for this coordinate
    bottom_coord_data = bottom_view_ds.position.sel(space=space_coord)
    triang_coord_data = triang_ds.position.sel(space=space_coord)
    bottom_min = np.min(triang_coord_data.values)
    
    # Scale to match triangulated data range
    bottom_range = np.ptp(bottom_view_ds.position.sel(space=space_coord).values)
    triang_range = np.ptp(triang_ds.position.sel(space=space_coord).values)
    if bottom_range > 0:  # Avoid division by zero
        scale_factor = triang_range / bottom_range
        bottom_view_ds.position.loc[dict(space=space_coord)] *= scale_factor
        bottom_view_ds.position.loc[dict(space=space_coord)] += bottom_min

# %%
plt.figure()
time_int_slice = slice(0, 10000)
for kp in "hindpaw_lf", "hindpaw_rt":
    for ds in [bottom_view_ds, triang_ds]:
        plt.scatter(ds.position.sel(keypoints=kp, space="x", time=time_int_slice), 
                    ds.position.sel(keypoints=kp, space="y", time=time_int_slice), 
                    s=10)
plt.show()

# %%

# Function to compute angle between two vectors
def compute_angle_between_vectors(v1, v2):
    """Compute angle between two vectors in degrees"""
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1, axis=0)
    v2_norm = v2 / np.linalg.norm(v2, axis=0)
    
    # Compute dot product
    dot_product = np.sum(v1_norm * v2_norm, axis=0)
    
    # Clip to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Compute angle in degrees
    angles = np.arccos(dot_product) * 180 / np.pi
    
    return angles

# Check available keypoints
print("Available keypoints in triangulated data:")
print(list(triang_ds.keypoints.values))

# Compute angles between the two lines for both datasets
datasets = {"triangulated": triang_ds, "bottom_view": bottom_view_ds}
results = {}

for ds_name, ds in datasets.items():
    # Get coordinates for line 1: hindpaw_lf to hindpaw_rt
    hindpaw_lf = ds.position.sel(keypoints="hindpaw_lf", space=["x", "y"]).values.squeeze()
    hindpaw_rt = ds.position.sel(keypoints="hindpaw_rt", space=["x", "y"]).values.squeeze()
    line1_vector = hindpaw_rt - hindpaw_lf
    
    # Get coordinates for line 2: tailbase to belly_caudal
    tailbase = ds.position.sel(keypoints="tailbase", space=["x", "y"]).values.squeeze()
    belly_caudal = ds.position.sel(keypoints="belly_caudal", space=["x", "y"]).values.squeeze()
    line2_vector = belly_caudal - tailbase
    print(line2_vector.shape, line1_vector.shape)
    # Compute angles over time
    angles = compute_angle_between_vectors(line1_vector.T, line2_vector.T)
    results[ds_name] = angles
    
    print(f"\n{ds_name} dataset:")
    print(f"Mean angle: {np.nanmean(angles):.2f}°")
    print(f"Std angle: {np.nanstd(angles):.2f}°")
    print(f"Min angle: {np.nanmin(angles):.2f}°")
    print(f"Max angle: {np.nanmax(angles):.2f}°")


# Plot angles over time
plt.figure(figsize=(12, 6))
time_values = triang_ds.time.values

for ds_name, angles in results.items():
    if angles is not None:
        plt.plot(time_values[:len(angles)], angles, label=f"{ds_name}", alpha=0.8)

plt.xlabel("Time")
plt.ylabel("Angle (degrees)")
plt.title("Angle between hindpaw line and tailbase-belly line over time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
angles.shape
# %%
