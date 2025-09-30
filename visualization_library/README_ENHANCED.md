# Enhanced Multi-Animal Arena Visualizer

A comprehensive visualization system for 3D animal tracking data with multi-animal support, statistics, and advanced visualization features.

## Features

### 🐭 Multi-Animal Support
- **Automatic animal detection**: Identifies mice, crickets, roaches, and other animals
- **Mixed animal types**: Visualize different animals in the same arena
- **2D/3D visualization**: 3D for mice, 2D flat for insects
- **Individual tracking**: Track multiple animals simultaneously

### 📊 Advanced Statistics
- **Velocity analysis**: Mean, max, and total velocity per animal
- **Distance calculations**: Inter-animal distances
- **Movement patterns**: Trajectory analysis and statistics
- **Frame-by-frame analysis**: Detailed movement metrics

### 🎯 Arena Integration
- **Proper arena positioning**: Corrected arena coordinate system
- **3D arena visualization**: Transparent mesh with edge lines
- **Coordinate alignment**: Ensures animals are properly positioned within arena
- **Multiple arena support**: Handle different arena configurations

### 🎨 Interactive Visualization
- **Plotly integration**: Zoom, pan, rotate, and interactive controls
- **Animation support**: Animated trajectories and movements
- **Multiple view modes**: 3D, 2D flat, trajectory, and statistics views
- **Export capabilities**: Save visualizations as HTML files

## Quick Start

### GUI Mode (Recommended)
```bash
python launch_enhanced_visualizer.py --gui
```

### CLI Mode
```bash
# Basic visualization
python launch_enhanced_visualizer.py --mouse-file data.h5 --arena-file arena.h5

# With statistics
python launch_enhanced_visualizer.py --mouse-file data.h5 --stats --frames 0 1000

# Save to file
python launch_enhanced_visualizer.py --mouse-file data.h5 --output visualization.html
```

## Usage Examples

### 1. Basic Multi-Animal Visualization
```python
from enhanced_visualizer import EnhancedMultiAnimalVisualizer

visualizer = EnhancedMultiAnimalVisualizer()

# Load data
visualizer.load_multi_animal_data("animal_data.h5")
visualizer.load_arena_data("arena_data.h5")

# Create plot
fig = visualizer.create_multi_animal_plot(frame_idx=0, show_arena=True)
fig.show()
```

### 2. Statistics Analysis
```python
# Calculate velocity statistics
stats = visualizer.calculate_velocity_statistics(0, 1000)

# Calculate distances between animals
distances = visualizer.calculate_distance_between_animals(frame_idx=0)

# Print summary
visualizer.print_statistics_summary()
```

### 3. Animation
```python
# Create animated plot
fig = visualizer.create_multi_animal_plot(
    frame_idx=0,
    show_arena=True,
    show_trajectory=True,
    trajectory_frames=100
)
```

## Animal Type Detection

The system automatically detects animal types based on:

### Mouse Detection
- **Keypoints**: `nose`, `ear_lf`, `ear_rt`, `back_rostral`, `back_mid`, `back_caudal`, `tailbase`
- **Coordinate ranges**: Small coordinate ranges (< 100mm)
- **Individual names**: Contains "mouse" or "rat"

### Insect Detection (Cricket/Roach)
- **Keypoints**: `head`, `thorax`, `abdomen`, `leg`
- **Individual names**: Contains "cricket", "roach", "insect"
- **2D visualization**: Flattened to arena floor

### Arena Detection
- **Large coordinates**: Coordinate ranges > 200mm
- **Few keypoints**: ≤ 10 keypoints
- **Individual names**: Contains "checkerboard", "arena"

## Statistics Features

### Velocity Metrics
- **Mean velocity**: Average movement per frame
- **Max velocity**: Peak movement speed
- **Total distance**: Cumulative movement distance
- **Standard deviation**: Velocity variability

### Distance Analysis
- **Inter-animal distances**: 3D distances between animals
- **Frame-by-frame tracking**: Distance changes over time
- **Proximity analysis**: Close encounters and interactions

### Movement Patterns
- **Trajectory analysis**: Path visualization and analysis
- **Speed profiles**: Velocity changes over time
- **Activity patterns**: Movement intensity analysis

## File Formats

### Input Files
- **H5 files**: Xarray-compatible HDF5 files
- **Structure**: `position(time, space, keypoints, individuals)`
- **Coordinates**: X, Y, Z in millimeters
- **Keypoints**: Anatomical landmarks
- **Individuals**: Animal identifiers

### Output Files
- **HTML**: Interactive Plotly visualizations
- **Statistics**: JSON-compatible data structures
- **Animations**: HTML5 video-compatible

## Troubleshooting

### Common Issues

#### 1. "No module named 'xarray'"
```bash
conda activate 3d_setup
```

#### 2. "Animals not visible"
- Check coordinate ranges
- Verify arena positioning
- Ensure proper keypoint detection

#### 3. "Arena positioning incorrect"
- Use the correct arena file
- Check coordinate system alignment
- Verify arena corner data

#### 4. "Statistics calculation fails"
- Ensure sufficient data frames
- Check for NaN values in coordinates
- Verify animal detection

### Debug Mode
```python
# Enable debug output
visualizer.debug_mode = True
visualizer.load_multi_animal_data("data.h5")
```

## Advanced Features

### Custom Skeletons
```python
# Define custom skeleton for new animal type
custom_skeleton = [
    ('head', 'neck'),
    ('neck', 'body'),
    # ... more connections
]

visualizer._get_animal_skeleton('custom_animal', custom_skeleton)
```

### Multi-File Analysis
```python
# Analyze multiple files
files = ['data1.h5', 'data2.h5', 'data3.h5']
for file in files:
    visualizer.load_multi_animal_data(file)
    stats = visualizer.calculate_velocity_statistics(0, 1000)
    print(f"File {file}: {stats}")
```

### Batch Processing
```python
# Process multiple frames
for frame in range(0, 1000, 100):
    fig = visualizer.create_multi_animal_plot(frame_idx=frame)
    fig.write_html(f"frame_{frame}.html")
```

## API Reference

### EnhancedMultiAnimalVisualizer

#### Methods
- `load_multi_animal_data(file_path)`: Load animal data
- `load_arena_data(file_path)`: Load arena data
- `create_multi_animal_plot()`: Create visualization
- `calculate_velocity_statistics()`: Calculate movement stats
- `calculate_distance_between_animals()`: Calculate inter-animal distances
- `create_statistics_plot()`: Create statistics visualization

#### Properties
- `animals_data`: Dictionary of detected animals
- `animal_types`: Animal type classifications
- `statistics`: Calculated statistics
- `arena_coords`: Arena coordinate data

## Contributing

### Adding New Animal Types
1. Update `_detect_animal_type_from_data()` method
2. Add skeleton definition in `_get_animal_skeleton()`
3. Update visualization logic if needed
4. Add tests for new animal type

### Adding New Statistics
1. Implement calculation method
2. Add to `calculate_velocity_statistics()`
3. Update `create_statistics_plot()` method
4. Add to GUI interface

## License

This project is part of the 3D tracking setup for the Iurilli lab.
