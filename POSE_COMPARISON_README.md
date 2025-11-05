# 3D Pose Comparison Tools

This directory contains tools for comparing original vs filtered 3D pose data with interactive visualization.

## 🚀 Quick Start

### Complete Pipeline (Recommended)
```bash
python compare_poses.py --input /path/to/data.h5 --config /path/to/config.yaml
```

### Individual Tools
```bash
# Step 1: Apply filtering
python threed_utils/savgol_filter.py --input_path /path/to/data.h5 --config /path/to/config.yaml

# Step 2: Compare poses
python threed_utils/pose_comparison.py --original /path/to/data.h5 --filtered /path/to/filtered_data.h5 --config /path/to/config.yaml
```

## 📁 Files Overview

### Main Scripts
- **`compare_poses.py`** - Complete pipeline script (recommended)
- **`threed_utils/savgol_filter.py`** - Applies Savitzky-Golay + outlier filtering
- **`threed_utils/pose_comparison.py`** - 3D pose comparison viewer

### Key Features
- **Savitzky-Golay Filtering** - Smooths position data
- **Outlier Detection** - Removes anomalous keypoints based on skeleton distances
- **3D Interactive Visualization** - Side-by-side pose comparison with frame slider
- **Skeleton Visualization** - Shows bone connections between keypoints

## 🛠️ Usage Examples

### Basic Usage
```bash
python compare_poses.py --input data.h5 --config config.yaml
```

### With Specific Keypoints
```bash
python compare_poses.py --input data.h5 --config config.yaml -k nose tailbase ear_lf ear_rt
```

### With Custom Parameters
```bash
python compare_poses.py --input data.h5 --config config.yaml --max_frame 5000 --filter savgol --outlier_threshold 3.0
```

## 📊 What You'll See

### Interactive 3D Viewer
- **Left Panel**: Original pose (gray skeleton)
- **Right Panel**: Filtered pose (red skeleton)
- **Frame Slider**: Navigate through all frames
- **Mouse Controls**: Zoom, rotate, pan

### Key Differences
- **Smoothing**: Filtered poses are smoother due to Savitzky-Golay filtering
- **Outlier Removal**: Missing keypoints where outliers were detected
- **Skeleton Integrity**: See how missing keypoints affect bone connections

## ⚙️ Configuration

### YAML Config File
Your config file should contain a `skeleton` section with bone connections:
```yaml
skeleton:
  - [nose, ear_lf]
  - [nose, ear_rt]
  - [ear_lf, ear_rt]
  - [ear_lf, back_rostral]
  # ... more connections
```

### Command Line Options

#### `compare_poses.py`
- `--input` - Path to original data (.h5 file)
- `--config` - Path to YAML config with skeleton
- `-k, --keypoints` - Specific keypoints to visualize
- `--max_frame` - Maximum number of frames to process
- `--filter` - Filter type: `savgol` or `lowpass`
- `--outlier_threshold` - Outlier detection threshold (standard deviations)

#### `threed_utils/savgol_filter.py`
- `--input_path` - Path to input data
- `--filter` - Filter type: `savgol` or `lowpass`
- `--config` - Path to YAML config
- `--outlier_threshold` - Outlier detection threshold
- `--max_frame` - Maximum frames to process (-1 for all)

#### `threed_utils/pose_comparison.py`
- `--original` - Path to original data
- `--filtered` - Path to filtered data
- `--config` - Path to YAML config
- `-k, --keypoints` - Keypoints to visualize
- `--max_frame` - Maximum frames to display

## 🔧 Troubleshooting

### Common Issues

1. **"No common keypoints found"**
   - Check that keypoints exist in both original and filtered datasets
   - Use `--keypoints` to specify which keypoints to visualize

2. **"YAML file is empty or invalid"**
   - Ensure your config file has a `skeleton` section
   - Check YAML syntax

3. **"Filtered data file not found"**
   - Run the filtering step first
   - Check that the filtered file was created in `~/Downloads/`

4. **Performance issues with large datasets**
   - Use `--max_frame` to limit the number of frames
   - Consider processing in chunks

### Performance Tips

- **Large datasets**: Use `--max_frame 1000` for initial testing
- **Memory issues**: Process smaller frame ranges
- **Slow visualization**: Reduce the number of keypoints with `-k`

## 📈 Understanding the Results

### Filtering Effects
- **Savitzky-Golay**: Reduces noise, makes trajectories smoother
- **Outlier Detection**: Removes biologically implausible poses
- **Skeleton Integrity**: Shows how filtering affects bone connections

### Visualization Features
- **Gray skeleton**: Original data (may be noisy)
- **Red skeleton**: Filtered data (smoother, outliers removed)
- **Missing keypoints**: Shown as gaps in the skeleton
- **Frame navigation**: Use slider to see changes over time

## 🔬 Scientific Interpretation

### What to Look For
1. **Smoothing effectiveness**: Are trajectories less noisy?
2. **Outlier removal**: Are biologically implausible poses removed?
3. **Skeleton integrity**: Do bone connections make sense?
4. **Temporal consistency**: Are changes smooth over time?

### Quality Assessment
- **Good filtering**: Smooth trajectories, plausible poses, consistent skeleton
- **Over-filtering**: Too much smoothing, loss of biological detail
- **Under-filtering**: Still noisy, outliers remain

## 📝 Output Files

- **Filtered data**: Saved to `~/Downloads/filtered_[original_name].h5`
- **Visualization**: Opens in web browser (interactive)
- **Logs**: Console output shows filtering statistics

## 🤝 Contributing

To add new features or fix issues:
1. Modify the relevant script in `threed_utils/`
2. Test with your data
3. Update this README if needed

## 📚 Dependencies

- `xarray` - Data handling
- `plotly` - Interactive visualization
- `numpy` - Numerical operations
- `pyyaml` - YAML config parsing
- `scipy` - Savitzky-Golay filtering
