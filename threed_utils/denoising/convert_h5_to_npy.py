"""
Convert h5 pose file (xarray format) to numpy format for denoising.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from argparse import ArgumentParser


def h5_to_npy(h5_path, output_path=None, individual_idx=0):
    """
    Convert h5 pose file to numpy format (N, 3, 13, 1).
    
    Args:
        h5_path: Path to h5 file (xarray format)
        output_path: Optional output path for npy file
        individual_idx: Which individual to extract (default 0)
    
    Returns:
        poses: (N, 3, 13, 1) numpy array
    """
    ds = xr.open_dataset(h5_path)
    
    # Extract positions for first individual
    positions = ds.position.isel(individuals=individual_idx)  # (time, space, keypoints)
    
    # Convert to (time, 3, 13, 1) format
    poses = positions.values  # (time, 3, keypoints)
    poses = np.transpose(poses, (0, 1, 2))  # (time, 3, keypoints)
    poses = poses[..., np.newaxis]  # (time, 3, keypoints, 1)
    
    # Reorder to match expected keypoint order if needed
    # The model expects 13 keypoints in specific order
    if output_path:
        np.save(output_path, poses.astype(np.float32))
        print(f"Saved {poses.shape} poses to {output_path}")
    
    return poses


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input h5 file")
    parser.add_argument("--output", type=Path, default=None, help="Output npy file")
    parser.add_argument("--individual", type=int, default=0, help="Individual index")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}.npy"
    
    poses = h5_to_npy(args.input, args.output, args.individual)
    print(f"Converted: {args.input} -> {args.output}")
    print(f"Shape: {poses.shape} (frames, 3, keypoints, 1)")

