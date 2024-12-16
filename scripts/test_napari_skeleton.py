# %%
import napari
import numpy as np
import skan
from skimage.data import binary_blobs
from skimage.morphology import skeletonize
import scipy.ndimage as ndi

#%%
# Ue synthetic data
blobs = binary_blobs(64, volume_fraction=0.3, n_dim=3)
binary_skeleton = skeletonize(blobs)
skeleton = skan.Skeleton(binary_skeleton)

all_paths = [
    np.column_stack([skeleton.path_coordinates(i), np.arange(len(skeleton.path_coordinates(i)))])
    for i in range(skeleton.n_paths)
]

paths_table = skan.summarize(skeleton, separator='_')
paths_table['path_id'] = np.arange(skeleton.n_paths)

# Color by random path ID
paths_table['random_path_id'] = np.random.default_rng().permutation(skeleton.n_paths)

viewer = napari.Viewer(ndisplay=3)

skeleton_layer = viewer.add_shapes(
    all_paths,
    shape_type='path',
    properties=paths_table,
    edge_width=0.5,
    edge_color='random_path_id',
    edge_colormap='tab10',
)

viewer.camera.angles = (-30, 30, -135)
viewer.camera.zoom = 6.5
napari.utils.nbscreenshot(viewer)

# Color by skeleton ID
skeleton_layer.edge_color = 'skeleton_id'
# For now, we need to set the face color as well
skeleton_layer.face_color = 'skeleton_id'

viewer.camera.angles = (-30, 30, -135)
napari.utils.nbscreenshot(viewer)

# Color by numerical property (e.g., branch_distance)
skeleton_layer.edge_color = 'branch_distance'
skeleton_layer.edge_colormap = 'viridis'
# For now, we need to set the face color as well
skeleton_layer.face_color = 'branch_distance'
skeleton_layer.face_colormap = 'viridis'

viewer.camera.angles = (-30, 30, -135)

viewer.show()

# %%
