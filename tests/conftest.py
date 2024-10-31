"""Test configuration and fixtures for pytest.

This module provides fixtures for managing test assets and artifacts, including
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
    return Path(__file__).parent.parent.parent / "test" / "assets"


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


@pytest.fixture
def session_artifacts(request, video_sessions: list[Path]) -> dict[Path, Path]:
    """
    Create and manage artifacts directories for each video session.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest request object.
    video_sessions : list[Path]
        List of video session directories.

    Returns
    -------
    dict[Path, Path]
        Mapping of session directories to their artifact directories.
    """
    # Create artifact directories
    artifacts = {}
    for session_dir in video_sessions:
        artifact_dir = session_dir / f"artifacts_{request.node.name}"
        artifact_dir.mkdir(exist_ok=True)
        artifacts[session_dir] = artifact_dir

    yield artifacts

    # Cleanup unless --keep-artifacts is specified
    if not request.config.getoption("--keep-artifacts"):
        for artifact_dir in artifacts.values():
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)


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
