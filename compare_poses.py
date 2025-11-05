#!/usr/bin/env python3
"""
Complete Pose Comparison Pipeline
Applies filtering and creates 3D pose comparison viewer
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Complete Pose Comparison Pipeline")
    parser.add_argument("--input", required=True, help="Path to original data .h5 file")
    parser.add_argument("--config", required=True, help="Path to YAML config file with skeleton")
    parser.add_argument("-k", "--keypoints", nargs="+", default=None, help="Keypoints to visualize")
    parser.add_argument("--max_frame", type=int, default=1000, help="Maximum frame range")
    parser.add_argument("--filter", type=str, default="savgol", choices=["savgol", "lowpass"], help="Smoothing filter type")
    parser.add_argument("--outlier_threshold", type=float, default=3.0, help="Outlier threshold in standard deviations")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    print("="*70)
    print("POSE COMPARISON PIPELINE")
    print("="*70)
    print(f"Input data: {input_path}")
    print(f"Config file: {config_path}")
    print(f"Keypoints: {args.keypoints or 'All available'}")
    print(f"Max frames: {args.max_frame}")
    print(f"Filter: {args.filter}")
    print(f"Outlier threshold: {args.outlier_threshold}")
    print("="*70)
    
    # Step 1: Apply filtering
    print("\nSTEP 1: APPLYING FILTERING")
    print("-" * 40)
    
    filter_cmd = [
        "python", "threed_utils/savgol_filter.py",
        "--input_path", str(input_path),
        "--filter", args.filter,
        "--config", str(config_path),
        "--outlier_threshold", str(args.outlier_threshold),
        "--max_frame", str(args.max_frame)
    ]
    
    print(f"Running: {' '.join(filter_cmd)}")
    result = subprocess.run(filter_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Filtering failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return 1
    
    print("Filtering completed successfully!")
    
    # Step 2: Create pose comparison
    print("\nSTEP 2: CREATING POSE COMPARISON")
    print("-" * 40)
    
    # Determine filtered file path
    filtered_path = Path.home() / "Downloads" / f"filtered_{input_path.stem}.h5"
    
    if not filtered_path.exists():
        print(f"ERROR: Filtered file not found: {filtered_path}")
        return 1
    
    comparison_cmd = [
        "python", "threed_utils/pose_comparison.py",
        "--original", str(input_path),
        "--filtered", str(filtered_path),
        "--config", str(config_path),
        "--max_frame", str(args.max_frame)
    ]
    
    if args.keypoints:
        comparison_cmd.extend(["-k"] + args.keypoints)
    
    print(f"Running: {' '.join(comparison_cmd)}")
    print(f"Original data: {input_path}")
    print(f"Filtered data: {filtered_path}")
    print(f"Config file: {config_path}")
    
    result = subprocess.run(comparison_cmd)
    
    if result.returncode != 0:
        print(f"ERROR: Pose comparison failed with return code {result.returncode}")
        return 1
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("The 3D pose comparison viewer should now be open in your browser.")
    print("Use the slider to navigate through frames and compare original vs filtered poses.")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
