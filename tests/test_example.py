import pytest

import threed_utils.visualization as tdv


@pytest.mark.parametrize("temp_dir", ["temp_mouse_data_dir", "temp_calib_data_dir"])
def test_temp_dir_exists(request, temp_dir: str):
    """
    Assert that the temporary data directory exists.
    """
    dir_path = request.getfixturevalue(temp_dir)
    assert dir_path.exists(), f"Temporary data directory {temp_dir} does not exist."


@pytest.mark.parametrize("temp_dir", ["temp_mouse_data_dir", "temp_calib_data_dir"])
def test_temp_dir_contains_expected_files(request, temp_dir: str):
    """
    Assert that the temporary data directories contain expected files.
    """
    dir_path = request.getfixturevalue(temp_dir)
    assert (
        len(list(dir_path.glob("*"))) == 5
    ), f"Temporary data directory {temp_dir} does not contain the expected number of files."


# def test_video_processing():
#     video_location = "tests/data/test_video.mp4"

#     cropped_video_dict = crop_video(video_location)

#     assert cropped_video_dict.keys() == ["left", "right", "top", "bottom"]

#     assert cropped_video_dict["left"].shape == (10, 10)

#     assert np.allequal(
#         cropped_video_dict["left"][0, :10], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     )


def test_helloworld():
    assert tdv.test_helloworld() == "the usual"
