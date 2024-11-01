import pytest
from pathlib import Path


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
    assert len(list(dir_path.glob("*"))) == 5, f"Temporary data directory {temp_dir} does not contain the expected number of files."
