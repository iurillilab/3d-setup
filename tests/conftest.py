"""Test configuration and fixtures for pytest.

This module provides fixtures for managing test assets, including
automatic cleanup of generated files.
"""

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


@pytest.fixture(scope="session")
def temp_data_dir(assets_dir: Path, tmp_path_factory, request) -> Path:
    """
    Create a temporary directory by copying the contents of the data subfolder.

    Parameters
    ----------
    assets_dir : Path
        Path to the test assets directory.
    tmp_path_factory : pytest.TempPathFactory
        Factory for creating temporary paths.
    request : pytest.FixtureRequest
        Fixture to access command line options.

    Yields
    -------
    Path
        Path to the temporary data directory.
    """
    data_subfolder = assets_dir / "temp_test_data" / assets_dir.name
    temp_dir = tmp_path_factory.mktemp(data_subfolder.name)
    shutil.copytree(data_subfolder, temp_dir, dirs_exist_ok=True)
    keep_artifacts = request.config.getoption("--keep-artifacts")
    try:
        yield temp_dir
    finally:
        if not keep_artifacts:
            shutil.rmtree(temp_dir)


@pytest.fixture
def video_files(video_sessions: list[Path]) -> dict[Path, list[Path]]:
    """
    Get dictionary mapping session directories to their video files.

    Parameters
    ----------
    video_sessions : list[Path]
        List of video session directories.

    Returns
    -------
    dict[Path, list[Path]]
        Mapping of session directories to lists of video file paths.
    """
    return {session: sorted(session.glob("*.mp4")) for session in video_sessions}
