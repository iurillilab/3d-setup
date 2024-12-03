import numpy as np
from pykalman import KalmanFilter

import numpy as np
from pykalman import KalmanFilter

def kalman_smoother_with_nans(keypoint_positions, confidence_scores, dt=1.0, max_gap=5):
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
    
    for i in range(N):  # Process each keypoint independently
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
    T, N = 100, 1
    np.random.seed(42)
    
    # Simulate smooth trajectories with noise
    # Create smooth sinusoidal trajectories
    t = np.linspace(0, 10, T)
    true_positions = np.zeros((T, N, 2))
    for i in range(N):
        # Different frequencies and phases for each keypoint
        true_positions[:, i, 0] = 5 * np.sin(t + i * np.pi/3)  # x coordinate
        true_positions[:, i, 1] = 5 * np.cos(1.5 * t + i * np.pi/4)  # y coordinate
    noisy_positions = true_positions + np.random.randn(T, N, 2) * 0.5
    
    # Add some NaN points randomly
    nan_mask = np.random.random(size=(T, N)) < 0.05  # 5% of points will be NaN
    noisy_positions[nan_mask] = np.nan
    
    # Generate confidences, setting to 0 where we have NaNs
    confidences = np.random.uniform(1.0, 1.5, size=(T, N))
    confidences[nan_mask] = 0
    
    # Apply Kalman smoother
    smoothed_positions = kalman_smoother_with_nans(noisy_positions, confidences, 
                                                   dt=1.0, max_gap=5)
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    for i in range(N):
        f, ax = plt.subplots(gridspec_kw=dict(right=0.8, top=0.8))
        plt.plot(true_positions[:, i, 0], true_positions[:, i, 1], label="True", alpha=0.6)
        plt.plot(noisy_positions[:, i, 0], noisy_positions[:, i, 1], '-', alpha=0.2, color='gray')
        sc = plt.scatter(noisy_positions[:, i, 0], noisy_positions[:, i, 1], c=confidences[:, i],
                   cmap='Reds', label="Noisy", alpha=0.5, s=20)
        plt.plot(smoothed_positions[:, i, 0], smoothed_positions[:, i, 1], '-', label="Smoothed", linewidth=2)
        plt.legend()
        plt.title(f"Keypoint {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")

        # Add colorbar in inset
        ax_inset = plt.axes([0.85, 0.15, 0.03, 0.3])
        plt.colorbar(sc, cax=ax_inset, label='Confidence')
        plt.show()

        