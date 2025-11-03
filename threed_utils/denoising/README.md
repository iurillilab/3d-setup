# 3D Pose Denoising Module

This module contains models and training scripts for denoising 3D pose coordinates of mice moving in an arena.

## Main Files

### Core Models
- **`temporal_DAE.py`** - Temporal denoising autoencoder model
  - Uses transformer/LSTM/conv1d for temporal modeling
  - Takes temporal windows of poses as input
  - Main model currently in use

- **`DAE.py`** - Standard (non-temporal) denoising autoencoder
  - Baseline model for single-frame denoising
  - May be used for comparison

### Training Scripts
- **`train_temporal.py`** - **Main training script**
  - Trains temporal DAE models
  - Supports multiple temporal architectures (transformer, LSTM, conv1d)
  - Supports multiple masking strategies (zero, gaussian, etc.)
  - Usage:
    ```bash
    python train_temporal.py --data_path /path/to/data.npy \
        --epochs 100 --window_size 2 --temporal_model transformer \
        --mask_strategy zero --log_dir ./logs
    ```

### Utilities
- **`feature_extraction.py`** - Data preprocessing utility
  - Extracts subsets of frames based on confidence scores
  - Selects high-quality frames for training
  - Usage:
    ```bash
    python feature_extraction.py --input /path/to/poses.nc --save
    ```

- **`convert_h5_to_npy.py`** - Convert h5 pose files to numpy format
  - Converts xarray h5 format to (N, 3, 13, 1) numpy arrays
  - Usage:
    ```bash
    python convert_h5_to_npy.py --input /path/to/poses.h5 --output /path/to/poses.npy
    ```

- **`visualization.py`** - Plotting utilities
  - 3D pose visualization
  - Keypoint error plots
  - Training progression plots
  - Used by training scripts for TensorBoard logging

### Documentation & Results
- **`RESEARCH_NOTES.md`** - Research notes and experimental plans
- **`grid_results.json`** - Grid search results (if available)

## Directory Structure

```
denoising/
├── temporal_DAE.py      # Main temporal model ⭐
├── train_temporal.py    # Main training script ⭐
├── constants.py         # Keypoint names & skeleton edges
├── visualization.py     # Plotting utilities
├── metrics.py           # Evaluation metrics
├── feature_extraction.py # Data preprocessing
├── convert_h5_to_npy.py # Convert h5 to numpy format
├── DAE.py               # Baseline model (for comparison)
├── README.md            # This file
└── RESEARCH_NOTES.md    # Research notes
```

## Key Features

### Temporal Models
- **Transformer**: Self-attention based temporal modeling
- **LSTM**: Bidirectional LSTM for temporal dependencies
- **Conv1D**: Convolutional temporal modeling

### Masking Strategies
- **zero**: Clean masking (sets masked keypoints to 0.0)
- **gaussian**: Adds Gaussian noise to unmasked keypoints
- More strategies may be available in the code

## Metrics

**`metrics.py`** - Comprehensive denoising evaluation metrics:
- RMSE, MAE
- Per-keypoint errors
- Masked vs unmasked breakdown
- Temporal smoothness
- Anatomical consistency

## Development Notes

- All plotting functions have been extracted to `visualization.py` to avoid duplication
- Constants (keypoint names, skeleton edges) are in `constants.py`
- TensorBoard logs are generated in `logs/` directory during training
- Models expect input shape: `(frames, 3, 13, 1)` for 13 keypoints in 3D space
- Training script includes comprehensive metrics logging every 10 epochs
- Legacy files (`train.py`, `DAE.py`) are kept for baseline comparison but not actively used

**TensorBoard Note:** If you see a KeyError, restart TensorBoard - this is typically a caching issue.

