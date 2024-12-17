"""Test configuration and fixtures for pytest.

This module provides fixtures for managing test assets, including
automatic cleanup of generated files.
"""

import pickle
import shutil
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep test artifacts after test completion",
    )


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    """
    Get the path to the test assets directory.

    Returns
    -------
    Path
        Path to the test assets directory.
    """
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def video_sessions(assets_dir: Path) -> list[Path]:
    """
    Get list of video session directories in assets.

    Parameters
    ----------
    assets_dir : Path
        Path to the test assets directory.

    Returns
    -------
    list[Path]
        List of paths to video session directories.
    """
    return [d for d in assets_dir.iterdir() if d.is_dir()]


def create_temp_dir(
    assets_dir: Path, tmp_path_factory, request, subfolder: str
) -> Path:
    """
    Create a temporary directory by copying the contents of the specified data subfolder.
    """
    data_subfolder = assets_dir / subfolder
    temp_dir = tmp_path_factory.mktemp(data_subfolder.name)
    shutil.copytree(data_subfolder, temp_dir, dirs_exist_ok=True)
    keep_artifacts = request.config.getoption("--keep-artifacts")
    if not keep_artifacts:
        request.addfinalizer(lambda: shutil.rmtree(temp_dir))

    return temp_dir


# @pytest.fixture(scope="session")
# def temp_mouse_data_dir(request, assets_dir, tmp_path_factory):
#     """Fixture to provide a temporary mouse data directory for testing."""
#     subfolder = "multicam_video_2024-07-24T10_20_07_cropped_20241031162801"
#     temp_dir = create_temp_dir(assets_dir, tmp_path_factory, request, subfolder)
#     return temp_dir


# mouse data dir 5 animals ds
# @pytest.fixture(scope="session")
# def triangulation_data_dir(request, assets_dir, tmp_path_factory):
#     """Fixture to provide temporary directory with triangulation test data."""
#     subfolder = "triangulation_test_data"
#     temp_dir = create_temp_dir(assets_dir, tmp_path_factory, request, subfolder)
#     return temp_dir

@pytest.fixture(scope="session")
def triangulation_data_dir(request, assets_dir, tmp_path_factory):
    """
    Fixture to provide a temporary directory with triangulation test data files.
    Creates a clean temporary directory containing only the required test files:
    - calibration.toml
    - arena_views.h5
    - arena_views_triangulated.h5
    """
    # Create base temporary directory
    temp_dir = tmp_path_factory.mktemp("triangulation_test_data")
    
    # Copy only the specific files we need
    required_files = [
        "calibration.toml",
        "arena_views.h5",
        "arena_views_triangulated.h5"
    ]
    
    for file in required_files:
        source = assets_dir / file
        destination = temp_dir / file
        shutil.copy2(source, destination)

    # Clean up temp directory after tests unless --keep-artifacts is used
    keep_artifacts = request.config.getoption("--keep-artifacts")
    if not keep_artifacts:
        request.addfinalizer(lambda: shutil.rmtree(temp_dir))

    return temp_dir


@pytest.fixture(scope="session")
def calibration_toml(triangulation_data_dir):
    """Fixture to provide path to calibration toml file."""
    return triangulation_data_dir / "calibration.toml"

@pytest.fixture(scope="session")
def arena_views_ds(triangulation_data_dir):
    """Fixture to provide input arena views dataset."""
    import xarray as xr
    return xr.open_dataset(triangulation_data_dir / "arena_views.h5")

@pytest.fixture(scope="session")
def expected_triangulated_ds(triangulation_data_dir):
    """Fixture to provide expected triangulated dataset."""
    import xarray as xr
    return xr.open_dataset(triangulation_data_dir / "arena_views_triangulated.h5")

# arena coordinates json


# @pytest.fixture(scope="session")
# def temp_calib_data_dir(request, assets_dir, tmp_path_factory):
#     """Fixture to provide a temporary calibration data directory for testing."""
#     subfolder = "multicam_video_2024-07-24T14_13_45_cropped_20241031162643"
#     temp_dir = create_temp_dir(assets_dir, tmp_path_factory, request, subfolder)
#     return temp_dir


# @pytest.fixture(scope="session")
# def coordinates_data_dict(assets_dir):
#     with open(assets_dir / "arena_tracked_points.pkl", "rb") as f:
#         data_dict = pickle.load(f)
#     return data_dict
#     # load in a dictionary
#     # return dictionary
