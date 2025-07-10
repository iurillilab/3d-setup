#%%
# %matplotlib widget
import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Configure backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from threed_utils.multiview_calibration.detection import run_checkerboard_detection, detect_chessboard, plot_chessboard_qc_data
from threed_utils.multiview_calibration.calibration import calibrate, get_intrinsics
from threed_utils.multiview_calibration.bundle_adjustment import bundle_adjust
from threed_utils.multiview_calibration.viz import plot_residuals, plot_shared_detections
from threed_utils.multiview_calibration.geometry import triangulate
from tqdm import tqdm, trange
import flammkuchen as fl
from threed_utils.io import write_calibration_toml
import cv2

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

@dataclass
class CalibrationOptions:
    board_shape: tuple[int, int] = (5, 7)
    square_size: float = 12.5
    scale_factor: float = 0.5
    n_workers: int = 6
    n_samples_for_intrinsics: int = 100
    ftol: float = 1e-4
    overlay: bool = False


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def find_video_files(data_dir: Path) -> List[Path]:
    """Find video files in the data directory."""
    video_paths = [
        f for f in data_dir.iterdir() 
        if f.suffix == ".mp4" and "overlay" not in f.stem
    ]
    
    if not video_paths:
        raise ValueError(f"No video files found in {data_dir}")
    
    return video_paths


def run_calibration(
    all_calib_uvs: np.ndarray,
    all_img_sizes: List[tuple],
    calib_objpoints: np.ndarray,
    options: CalibrationOptions,
) -> tuple:
    """Run the calibration process."""
    print("Running calibration...")
    
    all_extrinsics, all_intrinsics, calib_poses, spanning_tree = calibrate(
        all_calib_uvs,
        all_img_sizes,
        calib_objpoints,
        root=0,
        n_samples_for_intrinsics=options.n_samples_for_intrinsics,
    )
    
    return all_extrinsics, all_intrinsics, calib_poses, spanning_tree


def run_bundle_adjustment(
    all_calib_uvs: np.ndarray,
    all_extrinsics: np.ndarray,
    all_intrinsics: np.ndarray,
    calib_objpoints: np.ndarray,
    calib_poses: np.ndarray,
    options: CalibrationOptions,
) -> tuple:
    """Run bundle adjustment optimization."""
    print("Running bundle adjustment...")
    
    adj_extrinsics, adj_intrinsics, adj_calib_poses, use_frames, result = bundle_adjust(
        all_calib_uvs,
        all_extrinsics,
        all_intrinsics,
        calib_objpoints,
        calib_poses,
        n_frames=None,
        ftol=options.ftol,
    )
    
    return adj_extrinsics, adj_intrinsics, adj_calib_poses, use_frames, result


def save_results(
    output_dir: Path,
    all_calib_uvs: np.ndarray,
    all_img_sizes: List[tuple],
    calib_objpoints: np.ndarray,
    adj_extrinsics: np.ndarray,
    adj_intrinsics: np.ndarray,
    video_paths: List[Path],
    result: dict,
) -> None:
    """Save calibration results."""
    # Save numpy arrays
    np.save(output_dir / "all_calib_uvs.npy", all_calib_uvs)
    
    # Save calibration arguments
    fl.save(
        output_dir / "args_calibration.h5",
        dict(
            all_calib_uvs=all_calib_uvs,
            all_img_sizes=all_img_sizes,
            calib_objpoints=calib_objpoints,
        ),
    )
    
    # Save calibration to TOML
    cam_names = [Path(p).stem.split("_")[-1].split(".avi")[0] for p in video_paths]
    write_calibration_toml(
        output_dir / "calibration_from_mc.toml", 
        cam_names, 
        all_img_sizes, 
        adj_extrinsics, 
        adj_intrinsics, 
        result
    )


def generate_plots(
    output_dir: Path,
    video_paths: List[Path],
    all_calib_uvs: np.ndarray,
    all_extrinsics: np.ndarray,
    all_intrinsics: np.ndarray,
    calib_objpoints: np.ndarray,
    calib_poses: np.ndarray,
    adj_extrinsics: np.ndarray,
    adj_intrinsics: np.ndarray,
    adj_calib_poses: np.ndarray,
    use_frames: np.ndarray,
) -> None:
    """Generate and save calibration plots."""
    print("Generating plots...")
    
    # Plot corner-match scores
    fig = plot_chessboard_qc_data(video_paths)
    fig.savefig(output_dir / "checkerboard_errors.png")
    plt.close(fig)
    
    # Plot shared detections
    fig, shared_detections = plot_shared_detections(all_calib_uvs, spanning_tree)
    fig.savefig(output_dir / "shared_detections.png")
    plt.close(fig)
    
    # Plot residuals
    fig, median_error, reprojections, transformed_reprojections = plot_residuals(
        all_calib_uvs,
        all_extrinsics,
        all_intrinsics,
        calib_objpoints,
        calib_poses,
        inches_per_axis=3,
    )
    fig.savefig(output_dir / "first_residuals.png")
    plt.close(fig)
    
    # Plot refined residuals
    fig, median_error, reprojections, transformed_reprojections = plot_residuals(
        all_calib_uvs[:, use_frames],
        adj_extrinsics,
        adj_intrinsics,
        calib_objpoints,
        adj_calib_poses,
        inches_per_axis=3,
    )
    fig.savefig(output_dir / "refined_residuals.png")
    plt.close(fig)


def test_triangulation(
    output_dir: Path,
    all_calib_uvs: np.ndarray,
    adj_extrinsics: np.ndarray,
    adj_intrinsics: np.ndarray,
) -> None:
    """Test triangulation and save 3D plot."""
    print("Testing triangulation...")
    
    def triangulate_all_keypoints(
        calib_uvs, adj_extrinsics, adj_intrinsics, progress_bar=True
    ):
        all_triang = []
        progbar = tqdm if progress_bar else lambda x: x
        for i in progbar(range(calib_uvs.shape[2])):
            all_triang.append(
                triangulate(calib_uvs[:, :, i, :], adj_extrinsics, adj_intrinsics)
            )
        return np.array(all_triang)

    checkboard_3d = triangulate_all_keypoints(all_calib_uvs, adj_extrinsics, adj_intrinsics)

    non_nan_idxs = np.where(~np.isnan(checkboard_3d).any(axis=(0, 2)))[0]
    if len(non_nan_idxs) > 0:
        frame_idx = non_nan_idxs[0]
        checkboard_frame = checkboard_3d[:, frame_idx, :]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(checkboard_frame[:, 0], checkboard_frame[:, 1], checkboard_frame[:, 2])

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        plt.axis("equal")
        fig.savefig(output_dir / "triangulated_frame.png")
        plt.close(fig)


def run_calibration_pipeline(
    folder: Path,
    options: CalibrationOptions,
) -> None:
    """Run the complete calibration pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = folder / f"mc_calibration_output_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Find video files
    video_paths = find_video_files(folder)
    camera_names = [p.stem.split("_")[-1].split(".avi.mp4")[0] for p in video_paths]
    print(f"Found cameras: {camera_names}")
    
    # Run detection
    all_calib_uvs, all_img_sizes = run_checkerboard_detection(video_paths, options)
    
    # Generate object points
    calib_objpoints = generate_chessboard_objpoints(options.board_shape, options.square_size)
    
    # Run calibration
    all_extrinsics, all_intrinsics, calib_poses, spanning_tree = run_calibration(
        all_calib_uvs, all_img_sizes, calib_objpoints, options
    )
    
    # Run bundle adjustment
    adj_extrinsics, adj_intrinsics, adj_calib_poses, use_frames, result = run_bundle_adjustment(
        all_calib_uvs, all_extrinsics, all_intrinsics, calib_objpoints, calib_poses, options
    )
    
    # Generate plots
    generate_plots(
        output_dir, video_paths, all_calib_uvs, all_extrinsics, all_intrinsics,
        calib_objpoints, calib_poses, adj_extrinsics, adj_intrinsics, adj_calib_poses, use_frames
    )
    
    # Save results
    save_results(
        output_dir, all_calib_uvs, all_img_sizes, calib_objpoints,
        adj_extrinsics, adj_intrinsics, video_paths, result
    )
    
    # Test triangulation
    test_triangulation(output_dir, all_calib_uvs, adj_extrinsics, adj_intrinsics)
    
    print(f"Calibration complete! Results saved to: {output_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multicamera calibration pipeline.")
    parser.add_argument(
        "folder", 
        type=Path, 
        help="Directory containing video files for calibration."
    )
    parser.add_argument(
        "--board-shape", 
        type=int, 
        nargs=2, 
        default=CalibrationOptions.board_shape,
        help=f"Checkerboard shape (rows, cols). Default: {CalibrationOptions.board_shape}"
    )
    parser.add_argument(
        "--square-size", 
        type=float, 
        default=CalibrationOptions.square_size,
        help=f"Square size in mm. Default: {CalibrationOptions.square_size}"
    )
    parser.add_argument(
        "--scale-factor", 
        type=float, 
        default=CalibrationOptions.scale_factor,
        help=f"Scale factor for detection. Default: {CalibrationOptions.scale_factor}"
    )
    parser.add_argument(
        "--n-workers", 
        type=int, 
        default=CalibrationOptions.n_workers,
        help=f"Number of worker processes. Default: {CalibrationOptions.n_workers}"
    )
    parser.add_argument(
        "--n-samples-for-intrinsics",
        type=int,
        default=CalibrationOptions.n_samples_for_intrinsics,
        help=f"Number of samples for intrinsics. Default: {CalibrationOptions.n_samples_for_intrinsics}"
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=CalibrationOptions.ftol,
        help=f"Function tolerance for bundle adjustment. Default: {CalibrationOptions.ftol}"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Generate overlay videos (default: False)"
    )

    args = parser.parse_args()

    if not args.folder.exists():
        raise FileNotFoundError(f"Data directory not found: {args.folder}")

    options = CalibrationOptions(
        board_shape=tuple(args.board_shape),
        square_size=args.square_size,
        scale_factor=args.scale_factor,
        n_workers=args.n_workers,
        n_samples_for_intrinsics=args.n_samples_for_intrinsics,
        ftol=args.ftol,
        overlay=args.overlay,
    )

    run_calibration_pipeline(
        folder=args.folder,
        options=options,
    )
