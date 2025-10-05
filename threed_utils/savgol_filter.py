"""Script to apply Savitzky-Golay filter to a triangulations."""

from argparse import ArgumentParser
from os import wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pandas.errors import ParserError
from plotting_stats import plot_speed_subplots, plot_anchor_distances, plot_center_distances
from scipy.signal import butter, filtfilt, savgol_filter
from tqdm import tqdm


def denoise_positions_savgol(ds: xr.Dataset, window_s=0.5, polyorder=3):
    fps = 60
    window_length = int(window_s * fps)
    w = max(3, int(round(window_s * fps)))
    if w % 2 == 0:
        w += 1  # window length must be odd

    pos = ds["position"]
    arr = pos.values  # (T, S, K, I)
    arr_sm = savgol_filter(
        arr, window_length=w, polyorder=polyorder, axis=0, mode="interp"
    )
    return pos.copy(data=arr_sm)


def denoise_position_lowpass(ds: xr.Dataset, fc_hz=6.0, order=4):
    fps = 60
    wn = fc_hz / (fps / 2.0)  # Normalize the frequency
    b, a = butter(order, wn, btype="lowpass")
    pos = ds["position"].values  # (T, S, K, I)
    arr_sm = filtfilt(b, a, pos, axis=0, padlen=3 * max(len(a), len(b)) - 1)
    return ds["position"].copy(data=arr_sm)

def get_limb_outliers(ds: xr.Dataset, skeleton: list[tuple[str, str]], 
                     outlier_threshold: float = 3.0, 
                     min_frames_for_stats: int = 100):
    """
    Detect and mask outliers based on skeleton limb distances.
    
    For each skeleton connection, this function:
    1. Calculates the distance between connected keypoints over time
    2. Computes statistics (mean, std) for each limb distance
    3. Identifies frames where limb distances are outliers (beyond threshold * std)
    4. Masks those keypoints as NaN in the original dataset
    
    Args:
        ds: xarray Dataset with 'position' shaped (time, space, keypoints, individuals)
        skeleton: List of tuples defining skeleton connections [(kp1, kp2), ...]
        outlier_threshold: Number of standard deviations beyond which to consider outliers (default: 3.0)
        min_frames_for_stats: Minimum number of valid frames needed to compute statistics (default: 100)
    
    Returns:
        xr.Dataset: Copy of input dataset with outliers masked as NaN
    """
    import yaml
    import numpy as np
    from scipy import stats
    
    # Create a copy to avoid modifying the original
    ds_clean = ds.copy()
    
    # Get position data
    pos = ds_clean["position"]  # (time, space, keypoints, individuals)
    
    # Squeeze individuals dimension if it's size 1
    if "individuals" in pos.dims and pos.sizes.get("individuals", 1) == 1:
        pos = pos.squeeze("individuals")  # (time, space, keypoints)
    
    # Convert to numpy for easier manipulation
    pos_array = pos.values  # (T, S, K)
    
    print(f"Analyzing {len(skeleton)} skeleton connections for outliers...")
    print(f"Using outlier threshold: {outlier_threshold} standard deviations")
    
    # Track statistics for each limb
    limb_stats = {}
    outlier_keypoints = {}  # Track which keypoints are outliers in which frames
    
    for i, (kp1, kp2) in enumerate(skeleton):
        try:
            # Find keypoint indices
            kp1_idx = list(ds.keypoints.values).index(kp1)
            kp2_idx = list(ds.keypoints.values).index(kp2)
            
            # Calculate distances between these keypoints over time
            kp1_pos = pos_array[:, :, kp1_idx]  # (T, S)
            kp2_pos = pos_array[:, :, kp2_idx]  # (T, S)
            
            # Compute Euclidean distance for each frame
            distances = np.linalg.norm(kp1_pos - kp2_pos, axis=1)  # (T,)
            
            # Remove any NaN values for statistics calculation
            valid_distances = distances[~np.isnan(distances)]
            
            if len(valid_distances) < min_frames_for_stats:
                print(f"Warning: Not enough valid frames for {kp1}-{kp2} connection ({len(valid_distances)} < {min_frames_for_stats})")
                continue
            
            # Compute statistics
            mean_dist = np.mean(valid_distances)
            std_dist = np.std(valid_distances)
            
            # Store statistics
            limb_stats[f"{kp1}-{kp2}"] = {
                'mean': mean_dist,
                'std': std_dist,
                'distances': distances
            }
            
            # Find outliers
            outlier_mask = np.abs(distances - mean_dist) > outlier_threshold * std_dist
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(outlier_indices) > 0:
                print(f"Found {len(outlier_indices)} outlier frames for {kp1}-{kp2} connection")
                print(f"  Mean distance: {mean_dist:.2f} ± {std_dist:.2f}")
                print(f"  Outlier frames: {outlier_indices[:10]}{'...' if len(outlier_indices) > 10 else ''}")
                
                # Track which keypoints are outliers in which frames
                for frame_idx in outlier_indices:
                    if frame_idx not in outlier_keypoints:
                        outlier_keypoints[frame_idx] = set()
                    outlier_keypoints[frame_idx].add(kp1)
                    outlier_keypoints[frame_idx].add(kp2)
            
        except ValueError as e:
            print(f"Warning: Keypoint not found in dataset: {e}")
            continue
    
    # Apply masking to specific keypoints only
    total_outlier_frames = len(outlier_keypoints)
    total_keypoint_masks = 0
    
    if total_outlier_frames > 0:
        print(f"\nTotal frames with outliers: {total_outlier_frames}")
        
        # Mask only the specific keypoints that are outliers
        for frame_idx, outlier_kps in outlier_keypoints.items():
            for kp_name in outlier_kps:
                try:
                    kp_idx = list(ds.keypoints.values).index(kp_name)
                    pos_array[frame_idx, :, kp_idx] = np.nan
                    total_keypoint_masks += 1
                except ValueError:
                    continue
        
        # Update the dataset
        ds_clean["position"] = pos_array
        
        print(f"Masked {total_keypoint_masks} individual keypoint instances as outliers")
        print(f"Average keypoints masked per outlier frame: {total_keypoint_masks/total_outlier_frames:.1f}")
    else:
        print("No outliers detected!")
    
    # Print summary statistics
    print(f"\nLimb distance statistics:")
    for limb, stats in limb_stats.items():
        print(f"  {limb}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    return ds_clean


def load_skeleton_from_yaml(yaml_path: str) -> list[tuple[str, str]]:
    """
    Load skeleton connections from a YAML config file.
    
    Args:
        yaml_path: Path to the YAML config file
        
    Returns:
        List of tuples defining skeleton connections [(kp1, kp2), ...]
    """
    import yaml
    
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    skeleton = config.get('skeleton', [])
    
    # Convert list of lists to list of tuples
    skeleton_tuples = [tuple(connection) for connection in skeleton]
    
    print(f"Loaded {len(skeleton_tuples)} skeleton connections from {yaml_path}")
    print("Skeleton connections:")
    for i, (kp1, kp2) in enumerate(skeleton_tuples):
        print(f"  {i+1:2d}. {kp1} -> {kp2}")
    
    return skeleton_tuples


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input  file containing the triangulation data.",
    )
    PARSER.add_argument("-k", "--keypoints", nargs="+", action="extend", default=None)
    PARSER.add_argument("--filter", type=str, default="savgol")
    PARSER.add_argument("--max_frame", type=int, default=1000, help="Maximum frame to process (use -1 for entire dataset)")
    PARSER.add_argument("--ymin", type=float, default=None, help="Minimum y-axis value for zooming")
    PARSER.add_argument("--ymax", type=float, default=None, help="Maximum y-axis value for zooming")
    PARSER.add_argument("--config", type=str, default=None, help="Path to YAML config file with skeleton")
    PARSER.add_argument("--outlier_threshold", type=float, default=3.0, help="Outlier threshold in standard deviations")
    PARSER.add_argument("--plot_outliers", action="store_true", help="Generate outlier analysis plots")
    args = PARSER.parse_args()
    INPUT_PATH = Path(args.input_path)
    filtering_method = args.filter
    data_original = xr.open_dataset(INPUT_PATH)
    data = data_original.copy()
    
    # Apply filtering first, then select keypoints
    if filtering_method == "savgol":
        data["position"] = denoise_positions_savgol(data)
    elif filtering_method == "lowpass":
        data["position"] = denoise_position_lowpass(data)
    
    # Select keypoints after filtering
    if args.keypoints is not None:
        if len(args.keypoints) == 1:
            data = data.sel(keypoints=args.keypoints[0])
            data_original = data_original.sel(keypoints=args.keypoints[0])
        else:
            data = data.sel(keypoints=args.keypoints)
            data_original = data_original.sel(keypoints=args.keypoints)
    
    # Ensure both datasets have the same structure
    if len(data.position.shape) != 4:
        data = data.expand_dims({"keypoints": 1})
    if len(data_original.position.shape) != 4:
        data_original = data_original.expand_dims({"keypoints": 1})
    
    # Determine max_frame - use entire dataset if -1
    total_frames = len(data_original.time)
    max_frame = args.max_frame if args.max_frame != -1 else total_frames
    print(f"Dataset has {total_frames} frames, processing {max_frame} frames")
    
    # Set y-axis limits if provided
    ylim = None
    if args.ymin is not None or args.ymax is not None:
        ylim = (args.ymin, args.ymax)
        print(f"Setting y-axis limits: {ylim}")
    
    # Apply outlier detection if config is provided
    data_outlier_filtered = None
    if args.config is not None:
        print("\n" + "="*60)
        print("APPLYING OUTLIER FILTERING")
        print("="*60)
        
        # Load skeleton from YAML
        skeleton = load_skeleton_from_yaml(args.config)
        
        # Apply outlier filtering
        data_outlier_filtered = get_limb_outliers(
            data_original, 
            skeleton, 
            outlier_threshold=args.outlier_threshold
        )
        
        # Generate outlier analysis plots if requested
        if args.plot_outliers:
            print("\nGenerating separate outlier filtering visualizations...")
            
            # Plot anchor distances
            plot_anchor_distances(
                data_original,
                data_outlier_filtered,
                keypoint_cols=args.keypoints,
                max_frame=max_frame,
                skeleton=skeleton
            )
            
            # Plot center distances  
            plot_center_distances(
                data_original,
                data_outlier_filtered,
                keypoint_cols=args.keypoints,
                max_frame=max_frame
            )
            
            # Generate main comparison plot (original vs outlier-filtered)
            print("\nGenerating main comparison plot...")
            plot_speed_subplots(
                data_original,
                keypoint_cols=args.keypoints,
                max_frame=max_frame,
                dataset2=data_outlier_filtered,
                dataset1_label="Original",
                dataset2_label="Outlier-Filtered",
                title="Speed Comparison: Original vs Outlier-Filtered",
                ylim=ylim,
            )
    
    # Generate final Savitzky-Golay comparison plot
    print("\nGenerating Savitzky-Golay comparison plot...")
    plot_speed_subplots(
        data_original,
        keypoint_cols=args.keypoints,
        max_frame=max_frame,
        dataset2=data,
        dataset1_label="Original",
        dataset2_label=f"Filtered ({filtering_method})",
        title=f"Speed Comparison: Original vs {filtering_method.title()} Filtered",
        ylim=ylim,
    )
