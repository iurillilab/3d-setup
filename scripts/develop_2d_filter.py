import numpy as np
from tqdm import tqdm
from pykalman import KalmanFilter


def smooth_interpolation(keypoint_positions, max_gap=2, iterations=1, verbose=False):
    """
    Interpolate small gaps (up to `max_gap` frames) in keypoint trajectories while preserving smoothness.

    Parameters:
    - keypoint_positions: numpy array of shape (T, N, 2), where T is the number of time steps,
      N is the number of keypoints, and 2 represents (x, y) coordinates.
      Missing values are NaN.
    - max_gap: int, maximum consecutive NaN frames to interpolate.

    Returns:
    - interpolated_positions: numpy array of shape (T, N, 2), with small gaps filled.
    """
    T, N, D = keypoint_positions.shape
    interpolated_positions = np.copy(keypoint_positions)
    for i in tqdm(range(N), desc="Interpolating: ", disable=not verbose):  # Process each keypoint independently
        for d in range(D):  # Process each dimension (x, y) separately
            data = interpolated_positions[:, i, d]

            for _ in range(iterations):
                # Find all NaN indices
                nan_indices = np.where(np.isnan(data))[0]
                # print(nan_indices)
                
                # Merge gaps separated by single valid values
                # merged_gaps = []
                start = 0
                while start < len(nan_indices):
                    end = start
                    # Find end of current gap
                    while end + 1 < len(nan_indices) and nan_indices[end + 1] == nan_indices[end] + 1:
                        end += 1
                        
                    # Check if next gap is separated by just one frame
                    next_start = end + 1
                    if (next_start < len(nan_indices) and 
                        nan_indices[next_start] == nan_indices[end] + 2):
                        # Merge with next gap by continuing search
                        end = next_start
                        while end + 1 < len(nan_indices) and nan_indices[end + 1] == nan_indices[end] + 1:
                            end += 1
                    
                    gap_start, gap_end = nan_indices[start], nan_indices[end]

                    # Only interpolate if the gap is small enough
                    if (gap_end - gap_start + 1) <= max_gap:
                        # print("Interpolating gap", gap_start, gap_end)
                        prev_idx = gap_start - 1
                        next_idx = gap_end + 1

                        if prev_idx >= 0 and next_idx < T:
                            # Compute velocities for smoothness
                            prev_velocity = data[prev_idx] - (data[prev_idx - 1] if prev_idx > 0 else data[prev_idx])
                            next_velocity = (data[next_idx] - data[next_idx + 1]
                                            if next_idx + 1 < T else data[next_idx])

                            # Linearly interpolate the positions while matching edge velocities
                            alpha = np.linspace(0, 1, gap_end - gap_start + 2)
                            interpolated_segment = (
                                (1 - alpha) * (data[prev_idx] + prev_velocity * alpha[0]) +
                                alpha * (data[next_idx] - next_velocity * (1 - alpha[-1]))
                            )
                            interpolated_positions[gap_start:gap_end + 1, i, d] = interpolated_segment[1:]
                    else:
                        pass
                        # print("Not interpolating gap", gap_start, gap_end)
                    # Move to next segment
                    start = end + 1

    return interpolated_positions


def kalman_smoother_with_nans(keypoint_positions, confidence_scores, dt=1.0, max_gap=2, verbose=True):
    """
    Apply Kalman smoothing to tracked keypoints, handling NaN values.

    Parameters:
    - keypoint_positions: numpy array of shape (T, N, 2), where T is the number of time steps,
      N is the number of keypoints, and 2 represents (x, y) coordinates. Missing values are NaN.
    - confidence_scores: numpy array of shape (T, N), confidence for each keypoint at each time step.
    - dt: float, time step between frames.
    - max_gap: int, maximum consecutive NaN frames to interpolate.

    Returns:
    - smoothed_positions: numpy array of shape (T, N, 2), smoothed keypoint coordinates.
    """
    T, N, _ = keypoint_positions.shape

    # first, interpolate small gaps using linear interpolation,
    # and set confidence to 0 for interpolated points
    confidence_scores[np.isnan(keypoint_positions[:, :, 0])] = 0
    keypoint_positions = smooth_interpolation(keypoint_positions, max_gap, verbose=verbose)
    
    # Define the state transition matrix (constant velocity model)
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0,  1, 0],
                  [0, 0,  0, 1]])
    
    # Observation model: we observe positions only (x, y)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    # Process noise covariance (uncertainty in dynamics)
    Q = np.eye(4) * 0.01
    
    # Placeholder for smoothed positions
    smoothed_positions = np.full_like(keypoint_positions, np.nan)
    
    for i in tqdm(range(N), desc="Kalman smoothing: ", disable=not verbose):  # Process each keypoint independently
        # Extract observations and confidence for this keypoint
        observations = keypoint_positions[:, i, :]  # Shape (T, 2)
        confidences = confidence_scores[:, i]       # Shape (T,)

        # Mask missing values (NaNs)
        valid_mask = ~np.isnan(observations[:, 0])  # NaNs in x mean the whole observation is missing
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:  # Skip if too few valid data points
            continue

        # Measurement noise covariance: higher confidence -> lower noise
        R_base = np.eye(2) * 0.1
        R = R_base / np.mean(confidences[valid_mask])  # Simplified scaling by confidence

        # Initialize Kalman filter
        kf = KalmanFilter(
            transition_matrices=A,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=np.array([observations[valid_indices[0], 0],  # x0
                                          observations[valid_indices[0], 1],  # y0
                                          0,                                  # vx0
                                          0]),                                # vy0
            initial_state_covariance=np.eye(4) * 0.1
        )
        
        # Fit and smooth
        smoothed_state_means, _ = kf.smooth(observations[valid_mask])

        # Fill in the smoothed positions, respecting NaNs for long gaps
        for j, idx in enumerate(valid_indices):
            if j == 0 or idx - valid_indices[j - 1] <= max_gap:
                smoothed_positions[idx, i, :] = smoothed_state_means[j, :2]
    
    return smoothed_positions






# Example usage
if __name__ == "__main__":
    # Simulated data: T=100 time steps, N=3 keypoints
    T, N = 100, 13
    np.random.seed(42)
    
    # Simulate smooth trajectories with noise
    # Create smooth sinusoidal trajectories
    t = np.linspace(0, 10, T)
    true_positions = np.zeros((T, N, 2))
    for i in range(N):
        # Different frequencies and phases for each keypoint
        true_positions[:, i, 0] = 5 * np.sin(t + i * np.pi/3)  # x coordinate
        true_positions[:, i, 1] = 5 * np.cos(1.5 * t + i * np.pi/4)  # y coordinate
    # true_positions *= 10
    noisy_positions = true_positions + np.random.randn(T, N, 2) * 0.5
    
    # Add some NaN points randomly
    nan_mask = np.random.random(size=(T, N)) < 0.05  # 5% of points will be NaN
    nan_mask[80:89, :] = True
    noisy_positions[nan_mask] = np.nan
    
    # Generate confidences, setting to 0 where we have NaNs
    confidences = np.random.uniform(1.0, 1.5, size=(T, N))
    confidences[nan_mask] = 0
    
    # Apply Kalman smoother
    print(noisy_positions.shape, confidences.shape, type(noisy_positions), type(confidences))

    smoothed_positions = kalman_smoother_with_nans(noisy_positions, confidences, 
                                                   dt=1.0, max_gap=5)
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    # for i in range(N):
    #     f, ax = plt.subplots(gridspec_kw=dict(right=0.8, top=0.8))
    #     plt.plot(true_positions[:, i, 0], true_positions[:, i, 1], label="True", alpha=0.6)
    #     plt.plot(noisy_positions[:, i, 0], noisy_positions[:, i, 1], '-', alpha=0.2, color='gray')
    #     sc = plt.scatter(noisy_positions[:, i, 0], noisy_positions[:, i, 1], c=confidences[:, i],
    #                cmap='Reds', label="Noisy", alpha=0.5, s=20)
    #     plt.plot(smoothed_positions[:, i, 0], smoothed_positions[:, i, 1], '-', label="Smoothed", linewidth=2)
    #     plt.legend()
    #     plt.title(f"Keypoint {i+1}")
    #     plt.xlabel("x")
    #     plt.ylabel("y")

    #     # Add colorbar in inset
    #     ax_inset = plt.axes([0.85, 0.15, 0.03, 0.3])
    #     plt.colorbar(sc, cax=ax_inset, label='Confidence')
    #     plt.show()

        
    # %%

    import threed_utils.movement_napari as mn
    from movement.io.load_poses import from_file
    from matplotlib import pyplot as plt
    from pathlib import Path
    import ocplot as ocp
    import numpy as np
    # %matplotlib widget

    plots = False

    # %%
    data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
    # data_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\sample_video_for_triangulation\multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
    video_path = data_path / "multicam_video_2024-07-24T10_04_55_mirror-bottom.avi.mp4"
    slp_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.slp"))  # data_path / "multicam_video_2024-07-24T10_04_55_cropped_20241104101620.slp"
    dlc_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.h5"))  # data_path / "multicam_video_2024-07-24T10_04_55_central.aviDLC_resnet50_labels.v001.pkg_converted2024-11-21shuffle1_50000.h5"

    ds_sleap_full = from_file(slp_inference_path, source_software="SLEAP")
    # ds_dlc = from_file(dlc_inference_path, source_software="DeepLabCut")

    # Apply Kalman smoother
    time_slice = slice(None, None)
    noisy_positions = ds_sleap_full.position.sel(individuals="individual_0", time=time_slice, drop=True).values
    confidences = ds_sleap_full.confidence.sel(individuals="individual_0", time=time_slice, drop=True).values
    print(noisy_positions.shape, confidences.shape, type(noisy_positions), type(confidences))
    print("Running Kalman smoother...")
    smoothed_positions = kalman_smoother_with_nans(noisy_positions, confidences, 
                                                   dt=1.0, max_gap=5)
    
    filtered_ds = ds_sleap_full.copy()
    filtered_ds.position.values = smoothed_positions[:, np.newaxis, :, :]

    viewer = napari.Viewer() 
    viewer.add_image(VideoReaderNP(str(video_path)))
    keys = [i for i in list(to_compare.values())[0].keypoints.values]
    tracks_on = False

    to_compare = dict(original=ds_sleap_full, filtered=filtered_ds)

    import napari
    from movement.filtering import filter_by_confidence, interpolate_over_time
    from threed_utils.movement_napari.layer_styles import TracksStyle, PointsStyle
    from threed_utils.movement_napari.utils import columns_to_categorical_codes
    from threed_utils.movement_napari.convert import ds_to_napari_tracks
    from napari_video.napari_video import VideoReaderNP

    selected_keys = ["nose"]
    for (ds_name, ds), cmap in zip(to_compare.items(), ["Blues", "Greens"]):
        for key in selected_keys:
            ds_filt = ds.sel(keypoints=[key], drop=False)

            track, props = ds_to_napari_tracks(ds_filt)
            props.face_colormap = cmap

            points_style = PointsStyle(name=f"confidence_{key} - {ds_name}", properties=props, size=3)
            points_style.face_color = _custom_colormap(cmap, props["confidence"], conf_threshold=conf_threshold)

            if tracks_on:   
                tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])
                tracks_style = TracksStyle(name=f"Tracks_{key} - {ds_filt}", properties=tracks_props, tail_width=2)
                tracks_style.set_color_by(prop="confidence", cmap="turbo")

                viewer.add_tracks(track, **tracks_style.as_kwargs())
                
            viewer.add_points(track[:, 1:], **points_style.as_kwargs())

    viewer.show()
