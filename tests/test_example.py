import pytest
from pathlib import Path


def test_temp_dir_exists(temp_mouse_data_dir: Path):
    """
    Assert that the temporary data directory exists.

    Parameters
    ----------
    temp_data_dir : Path
        Path to the temporary data directory.
    """
    assert temp_mouse_data_dir.exists(), "Temporary data directory does not exist."


def test_temp_dir_contains_expected_files(temp_mouse_data_dir: Path):
    """
    Assert that the temporary data directory contains expected files.

    Parameters
    ----------
    temp_data_dir : Path
        Path to the temporary data directory.
    """
    # Define expected files relative to the data subfolder
    expected_files = [
        "sample1.csv",
        "sample2.csv",
        "config.yaml",
    ]

    for file_name in expected_files:
        file_path = temp_data_dir / file_name
        assert file_path.exists(), f"Expected file {file_name} does not exist in temporary data directory."
