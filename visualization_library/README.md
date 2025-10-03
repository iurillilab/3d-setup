# 3D Mouse Arena Visualization Library

A comprehensive library for visualizing mouse tracking data within experimental arenas using interactive 3D plots with Plotly.

## Features

- **Interactive 3D Visualization**: Zoom, pan, and rotate 3D plots
- **Mouse Skeleton Visualization**: Anatomically correct mouse skeleton
- **Arena Context**: Show mouse position within experimental arena
- **Trajectory Analysis**: Visualize mouse movement over time
- **Animation Support**: Animated plots showing movement
- **Multi-view Plots**: Different camera angles simultaneously
- **Statistical Analysis**: Movement statistics and heatmaps
- **GUI Interface**: Easy-to-use graphical interface
- **Data Export**: Export plots and data in various formats

## Arena Coordinates Used

The library uses the following arena corner coordinates (in mm):

```
Corner 0: ( -137.46,    14.43,   696.82) mm
Corner 1: (  186.24,    12.59,   686.51) mm  
Corner 2: (  194.40,   -11.92,  1021.21) mm
Corner 3: ( -130.37,    -7.98,  1032.99) mm
Corner 4: ( -140.92,  -128.56,   679.72) mm
Corner 5: (  182.83,  -126.37,   674.99) mm
Corner 6: (  191.19,  -156.10,  1007.51) mm
Corner 7: ( -131.21,  -147.52,  1017.67) mm
```

**Arena Dimensions**: 335.3mm × 170.5mm × 358.0mm (width × length × height)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the required data files:
   - Mouse triangulation data (H5 format)
   - Arena data (H5 format)

## Usage

### GUI Application

Run the graphical interface:
```bash
python gui_visualizer.py
```

Features:
- File browser for mouse and arena data
- Interactive frame selection
- Real-time visualization options
- Data information display
- Export capabilities

### Programmatic Usage

```python
from mouse_arena_visualizer import MouseArenaVisualizer

# Initialize visualizer
viz = MouseArenaVisualizer()

# Load data
viz.load_mouse_data("path/to/mouse_data.h5")
viz.load_arena_data("path/to/arena_data.h5")

# Create interactive plot
fig = viz.create_interactive_plot(
    frame_idx=0,
    show_arena=True,
    show_trajectory=True,
    trajectory_frames=100
)

# Save plot
fig.write_html("output.html")
```

### Advanced Features

```python
from plotly_visualizer import AdvancedPlotlyVisualizer

# Initialize advanced visualizer
viz = AdvancedPlotlyVisualizer()

# Create animated plot
fig = viz.create_animated_plot(
    start_frame=0,
    end_frame=100,
    frame_step=5
)

# Create multi-view plot
fig = viz.create_multi_view_plot(frame_idx=50)

# Create heatmap
fig = viz.create_heatmap_plot(start_frame=0, end_frame=100)

# Create statistics plot
fig = viz.create_statistics_plot(start_frame=0, end_frame=100)
```

## File Structure

```
visualization_library/
├── __init__.py                 # Package initialization
├── mouse_arena_visualizer.py   # Core visualization class
├── gui_visualizer.py           # GUI application
├── plotly_visualizer.py        # Advanced Plotly features
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Mouse Skeleton

The library uses an anatomically correct mouse skeleton with the following connections:

- **Head**: nose ↔ ears, ears ↔ ears
- **Spine**: back_rostral → back_mid → back_caudal → tailbase
- **Belly**: belly_rostral ↔ belly_caudal
- **Limbs**: 
  - Forelimbs: forepaws ↔ back_rostral
  - Hindlimbs: hindpaws ↔ back_caudal
- **Connections**: nose ↔ back_rostral, belly ↔ spine

## Key Features

### Interactive 3D Plotting
- **Zoom**: Mouse wheel or pinch gestures
- **Pan**: Click and drag
- **Rotate**: Click and drag with right mouse button
- **Reset View**: Double-click to reset camera

### Arena Visualization
- **8-corner rectangular arena**
- **Transparent mesh** showing arena boundaries
- **Edge lines** highlighting arena structure
- **Coordinate system** in millimeters

### Mouse Tracking
- **13 keypoints** with anatomical names
- **Skeleton connections** showing body structure
- **Trajectory visualization** showing movement paths
- **Frame-by-frame analysis**

### Export Options
- **HTML plots** for web viewing
- **JSON data export** for analysis
- **Statistical summaries** of movement
- **Multi-format support**

## Examples

### Basic Visualization
```python
# Load data and create basic plot
viz = MouseArenaVisualizer()
viz.load_mouse_data("mouse_data.h5")
viz.load_arena_data("arena_data.h5")
fig = viz.create_interactive_plot(frame_idx=0, show_arena=True)
fig.show()
```

### Trajectory Analysis
```python
# Show mouse movement over time
fig = viz.create_interactive_plot(
    frame_idx=100,
    show_arena=True,
    show_trajectory=True,
    trajectory_frames=50
)
fig.show()
```

### Animation
```python
# Create animated plot
advanced_viz = AdvancedPlotlyVisualizer()
fig = advanced_viz.create_animated_plot(
    start_frame=0,
    end_frame=200,
    frame_step=5
)
fig.show()
```

## Troubleshooting

### Common Issues

1. **File not found**: Ensure H5 files exist and paths are correct
2. **Memory issues**: Reduce trajectory_frames for large datasets
3. **Slow rendering**: Use frame_step > 1 for animations
4. **Missing dependencies**: Install requirements.txt

### Performance Tips

- Use `frame_step > 1` for animations with many frames
- Reduce `trajectory_frames` for large datasets
- Close other applications when processing large files
- Use SSD storage for better I/O performance

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify file formats and paths
3. Ensure all dependencies are installed
4. Check data file integrity

## License

This library is part of the 3D tracking setup project.
