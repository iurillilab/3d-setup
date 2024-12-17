import numpy as np
import pytest

# Now import your function
#%%
from threed_utils.triangulation_ds import mcc_triangulate_ds

from threed_utils.old_triangulation import triangulate_all_keypoints
import numpy as np
import xarray as xr
from pathlib import Path
from threed_utils.triangulation_ds import mcc_triangulate_ds 
#%%
# @pytest.mark.parameterize("alpha, expected_output", [(0.5, 0.1), (1, 5)]
def test_arena_triangulation(
    arena_views_ds,
    calibration_toml,
    expected_triangulated_ds
):
    """Test that triangulation produces expected 3D coordinates for arena points."""
    
    # Run triangulation
    actual_ds = mcc_triangulate_ds(arena_views_ds, calibration_toml)
    
    # Compare position values
    np.testing.assert_allclose(
        actual_ds.position.values,
        expected_triangulated_ds.position.values,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Triangulated positions don't match expected values"
    )
