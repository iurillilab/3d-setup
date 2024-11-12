import numpy as np
import pytest

from threed_utils.triangulation import triangulate_all_keypoints

# @pytest.mark.parameterize("alpha, expected_output", [(0.5, 0.1), (1, 5)])
# def triangulate(data_dictionary, alpha, expected_output):
#     out = triangulate(..., alpha)

#     assert out == expected_output


# def output_triangulation(data_dict):
#     #for key, value in data_dict.items():

#     all_triang = triangulate_all_keypoints(data_dict[data...], data_dict["adj_extrinsics"]...)

#     assert all_triang.shape[2] == 3
#     # np.allclose to check correctedness of the output
