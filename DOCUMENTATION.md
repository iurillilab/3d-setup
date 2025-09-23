# 3D Tracking Setup - Complete Documentation

## Overview

This repository contains a comprehensive pipeline for 3D pose estimation and tracking using a multi-camera mirror setup. The system processes video data from multiple camera views to reconstruct 3D coordinates of animal poses, with particular focus on mouse behavior analysis in hunting scenarios.

## Project Structure

```
3d-setup/
├── complete_pipeline/          # Main processing pipeline scripts
├── threed_utils/              # Core utility modules
├── scripts/                   # Additional processing scripts
├── notebooks/                 # Analysis and debugging notebooks
├── tests/                     # Test files and sample data
├── build/                     # Build artifacts
└── requirements.txt           # Python dependencies
```

## Core Pipeline

The processing pipeline consists of four main stages:

### Stage 0: Video Preparation and Cropping
- **Scripts**: `0a_define_cropping.py`, `0b_process_videos.py`
- **Purpose**: Interactive cropping of multi-view videos using Napari GUI
- **Output**: Cropped video files with defined regions of interest

### Stage 1: Checkerboard Detection and Calibration
- **Scripts**: `1_extract_checkerboards.py`, `2_calibration_multicam_script.py`
- **Purpose**: Extract checkerboard patterns and perform multi-camera calibration
- **Output**: Camera calibration parameters and intrinsic/extrinsic matrices

### Stage 2: 3D Triangulation
- **Scripts**: `3_triangulation_multicam_anipose_script.py`, `3_triangulation.py`
- **Purpose**: Triangulate 2D keypoints to 3D coordinates using Anipose
- **Output**: 3D pose data in movement format

### Stage 3: Data Analysis and Visualization
- **Tools**: Various notebooks and visualization scripts
- **Purpose**: Analyze 3D trajectories, generate plots, and validate results

## Key Modules

### `threed_utils/` - Core Utilities

#### `io.py`
- **Purpose**: Data input/output operations
- **Key Functions**:
  - `movement_ds_from_anipose_triangulation_df()`: Convert Anipose triangulation to movement dataset
  - `read_calibration_toml()`: Load camera calibration parameters
  - `write_calibration_toml()`: Save calibration parameters

#### `arena_utils.py`
- **Purpose**: Arena triangulation and visualization
- **Key Functions**:
  - `load_arena_coordinates()`: Load arena coordinate definitions
  - `load_arena_multiview_ds()`: Create movement dataset from arena coordinates
  - `triangulate_arena_points()`: Triangulate arena reference points

#### `anipose/` - Anipose Integration
- **`triangulate.py`**: Core triangulation functions
- **`calibrate.py`**: Camera calibration utilities
- **`anipose_filtering_2d.py`**: 2D filtering and preprocessing
- **`movement_anipose.py`**: Integration with movement library

#### `multiview_calibration/` - Multi-camera Calibration
- **`detection.py`**: Checkerboard detection algorithms
- **`calibration.py`**: Camera calibration procedures
- **`bundle_adjustment.py`**: Bundle adjustment optimization
- **`geometry.py`**: Geometric transformations
- **`viz.py`**: Calibration visualization

#### `movement_napari/` - Napari Integration
- **Purpose**: Interactive visualization and analysis in Napari
- **Features**: Layer styles, loader widgets, metadata widgets

### `scripts/` - Additional Processing Tools

#### `anipose/` - Anipose Processing
- **`run_2d_filter.py`**: 2D filtering pipeline
- **`check_triangulation.py`**: Triangulation validation
- **`test_arena_triangulation.py`**: Arena triangulation testing

#### `dlc/` - DeepLabCut Integration
- **`run_dlc_inference.py`**: DLC inference execution
- **`run_dlc_training.py`**: DLC model training
- **`convert_labels_sleap2dlc.py`**: Format conversion utilities

#### `sleap/` - SLEAP Integration
- **`model_inference.py`**: SLEAP model inference
- **`sleap_training.py`**: SLEAP model training
- **`crop_and_inference.py`**: Cropped video inference

## Configuration

### Pipeline Parameters (`pipeline_params.py`)

```python
@dataclass
class CroppingOptions:
    crop_folder_pattern: str = "cropped-v2"
    expected_views: tuple[str] = ("central", "mirror-bottom", "mirror-left", "mirror-right", "mirror-top")

@dataclass
class DetectionOptions:
    board_shape: tuple[int, int] = (5, 7)
    match_score_min_diff: float = 0.15
    match_score_min: float = 0.4

@dataclass
class CalibrationOptions:
    square_size: float = 12.5
    scale_factor: float = 0.5
    n_samples_for_intrinsics: int = 100
    ftol: float = 1e-4
```

### Dependencies

**Core Requirements** (`requirements.txt`):
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `napari`: Interactive visualization
- `movement`: Pose data handling
- `PyYAML`: Configuration files
- `vidio`: Video I/O
- `flammkuchen`: Data serialization
- `toml`: TOML file handling
- `aniposelib`: 3D pose estimation
- `hickle`: HDF5 serialization

## Detailed Script Usage and Purpose

### Complete Pipeline Scripts (`complete_pipeline/`)

#### Stage 0: Video Preparation
**`0a_define_cropping.py`**
- **Purpose**: Interactive definition of cropping windows using Napari GUI
- **Usage**: `python 0a_define_cropping.py /path/to/movie.avi`
- **Output**: JSON file with FFmpeg cropping parameters
- **Features**: 
  - Interactive window selection for 5 camera views (central, mirror-top, mirror-bottom, mirror-left, mirror-right)
  - Real-time preview of cropping results
  - Automatic transformation filters for mirror views

**`0b_process_videos.py`**
- **Purpose**: Batch processing of videos using defined cropping parameters
- **Usage**: `python 0b_process_videos.py /path/to/videos /path/to/cropping_parameters.json`
- **Features**:
  - Processes all AVI files in specified directory
  - Skips already processed files (with `_cropped_` suffix)
  - Applies view-specific transformations (mirror flips, rotations)
  - Generates timestamped output folders

**`0c_opt_check_video_integrity.py`**
- **Purpose**: Validate cropped videos against original frame counts
- **Usage**: `python 0c_opt_check_video_integrity.py /path/to/folder [options]`
- **Features**:
  - Compares frame counts between original and cropped videos
  - Reports mismatches, missing files, and errors
  - Generates integrity reports

#### Stage 1: Checkerboard Detection
**`1_extract_checkerboards.py`**
- **Purpose**: Extract checkerboard patterns from calibration videos
- **Usage**: `python 1_extract_checkerboards.py /path/to/calibration/videos [options]`
- **Options**:
  - `--board-shape`: Checkerboard dimensions (default: 5,7)
  - `--match-score-min`: Minimum detection confidence (default: 0.4)
  - `--video-extension`: Video file extension (default: mp4)
- **Output**: Detection results saved as HDF5 files

#### Stage 2: Multi-camera Calibration
**`2_calibration_multicam_script.py`**
- **Purpose**: Perform multi-camera calibration using checkerboard detections
- **Usage**: `python 2_calibration_multicam_script.py /path/to/detections`
- **Features**:
  - Intrinsic and extrinsic camera parameter estimation
  - Bundle adjustment optimization
  - Calibration quality assessment
  - TOML format calibration output

#### Stage 3: 3D Triangulation
**`3_triangulation_multicam_anipose_script.py`**
- **Purpose**: Triangulate 2D poses to 3D coordinates using Anipose
- **Usage**: `python 3_triangulation_multicam_anipose_script.py /path/to/2d/poses /path/to/calibration`
- **Features**:
  - Multi-camera triangulation with confidence scoring
  - 2D filtering and outlier detection
  - 3D pose reconstruction
  - Movement dataset output

**`3_triangulation.py`**
- **Purpose**: Simplified triangulation workflow
- **Usage**: `python 3_triangulation.py`
- **Features**: Streamlined triangulation for testing and development

### Utility Scripts (`scripts/`)

#### DeepLabCut Integration (`scripts/dlc/`)
**`run_dlc_inference.py`**
- **Purpose**: Run DeepLabCut inference on video folders
- **Usage**: `python run_dlc_inference.py config.yaml /path/to/videos [options]`
- **Options**:
  - `--make-labeled-video`: Generate labeled output videos
  - `--shuffle-n`: Shuffle index (default: 2)
  - `--batch-size`: Inference batch size (default: 2)

**`run_dlc_training.py`**
- **Purpose**: Train DeepLabCut models
- **Usage**: `python run_dlc_training.py config.yaml`
- **Features**: Automated model training pipeline

**`convert_labels_sleap2dlc.py`**
- **Purpose**: Convert SLEAP labels to DeepLabCut format
- **Usage**: `python convert_labels_sleap2dlc.py /path/to/sleap/labels /path/to/output`

#### SLEAP Integration (`scripts/sleap/`)
**`model_inference.py`**
- **Purpose**: Run SLEAP model inference on videos
- **Usage**: `python model_inference.py /path/to/videos`
- **Features**:
  - Automatic video discovery
  - Batch processing with progress tracking
  - Side view and bottom view model support

**`sleap_training.py`**
- **Purpose**: Train SLEAP models
- **Usage**: `python sleap_training.py config.json`

#### Anipose Processing (`scripts/anipose/`)
**`run_2d_filter.py`**
- **Purpose**: Apply 2D filtering to pose data
- **Usage**: `python run_2d_filter.py /path/to/poses`
- **Features**: Confidence-based filtering and outlier removal

**`check_triangulation.py`**
- **Purpose**: Validate triangulation results
- **Usage**: `python check_triangulation.py /path/to/triangulated/data`
- **Features**: Quality assessment and visualization

#### Visualization and Analysis
**`backprojection_gen.py`**
- **Purpose**: Generate 2D backprojection visualizations
- **Usage**: `python backprojection_gen.py /path/to/3d/data /path/to/calibration`
- **Features**: 3D to 2D projection for validation

**`backproject_2_napari_plugin.py`**
- **Purpose**: Napari plugin for backprojection visualization
- **Usage**: Load as Napari plugin
- **Features**: Interactive 3D pose visualization

#### Data Processing
**`merging_videos.py`** (in `scripts/lighting/`)
- **Purpose**: Merge multiple video files
- **Usage**: `python merging_videos.py /path/to/videos /path/to/output`
- **Features**: Video concatenation and synchronization

**`reencode_h264.py`**
- **Purpose**: Re-encode videos to H.264 format
- **Usage**: `python reencode_h264.py /path/to/input /path/to/output`
- **Features**: Format conversion and compression

### Analysis Notebooks (`notebooks/`)

#### Debugging and Validation
- **`debug_coordinates_transformation.ipynb`**: Coordinate system debugging
- **`debug_cropping.ipynb`**: Cropping validation and visualization
- **`debug_detection.ipynb`**: Detection quality assessment
- **`debug_triangulation_files.ipynb`**: Triangulation validation

#### Data Analysis
- **`data_analysis_mergining_predictions.ipynb`**: Prediction merging and analysis
- **`filtering.ipynb`**: Data filtering techniques
- **`dlc_plots.ipynb`**: DeepLabCut result visualization
- **`sleap_plots.ipynb`**: SLEAP result visualization

#### Calibration and Setup
- **`open_calibration.ipynb`**: Calibration data inspection
- **`2a_check_calibration.ipynb`**: Calibration validation
- **`back_projection.ipynb`**: Backprojection analysis

### Usage Workflows

#### Complete Processing Pipeline
```bash
# 1. Define cropping parameters
python 0a_define_cropping.py /path/to/calibration_video.avi

# 2. Process all videos
python 0b_process_videos.py /path/to/videos /path/to/cropping_params.json

# 3. Check video integrity
python 0c_opt_check_video_integrity.py /path/to/processed/videos

# 4. Extract checkerboards
python 1_extract_checkerboards.py /path/to/calibration/videos

# 5. Perform calibration
python 2_calibration_multicam_script.py /path/to/checkerboard/detections

# 6. Run pose detection (SLEAP or DLC)
python scripts/sleap/model_inference.py /path/to/cropped/videos

# 7. Triangulate to 3D
python 3_triangulation_multicam_anipose_script.py /path/to/2d/poses /path/to/calibration
```

#### Individual Component Usage
```bash
# Run DLC inference
python scripts/dlc/run_dlc_inference.py config.yaml /path/to/videos --make-labeled-video

# Apply 2D filtering
python scripts/anipose/run_2d_filter.py /path/to/poses

# Generate backprojections
python scripts/backprojection_gen.py /path/to/3d/data /path/to/calibration

# Check triangulation quality
python scripts/anipose/check_triangulation.py /path/to/triangulated/data
```

## Data Formats

### Input Data
- **Videos**: Multi-view AVI/MP4 files with synchronized cameras
- **Calibration**: Checkerboard calibration videos
- **2D Poses**: SLEAP or DeepLabCut output files (.slp, .h5)

### Output Data
- **3D Poses**: Movement-format datasets with 3D coordinates
- **Calibration**: TOML files with camera parameters
- **Visualizations**: Plots and videos with 3D trajectories

### Keypoint Schema
The system uses a standardized keypoint schema for mouse poses:
- **Head**: nose, ear_lf, ear_rt
- **Body**: back_rostral, back_mid, back_caudal, belly_rostral, belly_caudal
- **Limbs**: forepaw_lf, forepaw_rt, hindpaw_lf, hindpaw_rt
- **Tail**: tailbase

## Visualization Tools

### Napari Integration
- Interactive 3D pose visualization
- Multi-layer data overlay
- Real-time parameter adjustment
- Export capabilities

### Plotting Utilities
- **`frame_plots.py`**: Frame-by-frame visualization
- **`animation_tools.py`**: 3D animation generation
- **`backprojection.py`**: 2D backprojection visualization

## Testing

### Test Data
- Sample calibration files in `tests/assets/`
- Example video data for validation
- Reference triangulation results

### Test Scripts
- **`test_triangulation.py`**: Triangulation accuracy tests
- **`test_example.py`**: Basic functionality tests

## Development

### Installation
```bash
pip install -e .
```

### Development Dependencies
```bash
pip install -e .[dev]
```

### Code Quality
- Black formatting
- isort import sorting
- Type hints throughout

## File Organization

### Input Data Structure
```
data/
├── calibration/
│   ├── cropping_params.json
│   └── 20250509/
│       └── calibration session data
├── M29, M30, M31/
│   ├── cricket/
│   │   ├── *.csv               # timestamps
│   │   ├── *.avi               # original video
│   │   └── 133050/
│   │       ├── multicam_video_*_cropped-v2_*/
│   │       │   ├── cropped videos (.mp4)
│   │       │   └── Tracking/
│   │       │       ├── Central: full.pickle, meta.pickle, snapshot.h5
│   │       │       ├── Sides: full.pickle, meta.pickle, snapshot.h5
│   │       │       └── Triangulations: datetime.h5
│   └── object/
│       └── (same structure)
```

### Output Data Structure
```
output/
├── mc_calibration_output_*/
│   ├── all_calib_uvs.npy
│   ├── calibration_from_mc.toml
│   └── calibration_plots/
├── triangulation_results/
│   ├── 3d_poses.h5
│   └── validation_plots/
└── analysis_results/
    ├── trajectory_plots/
    └── statistics/
```

## Troubleshooting

### Common Issues
1. **Calibration Quality**: Ensure checkerboard is visible in all views
2. **Synchronization**: Verify camera synchronization
3. **Cropping Consistency**: Use same cropping for calibration and data videos
4. **Memory Usage**: Large videos may require chunked processing

### Debug Tools
- **`debug_coordinates_transformation.ipynb`**: Coordinate system debugging
- **`debug_cropping.ipynb`**: Cropping validation
- **`debug_detection.ipynb`**: Detection quality assessment
- **`debug_triangulation_files.ipynb`**: Triangulation validation



---

*This documentation was generated automatically from the codebase structure and content.*
