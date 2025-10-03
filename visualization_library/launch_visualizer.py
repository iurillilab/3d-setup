#!/usr/bin/env python3
"""
Main launcher for the 3D Mouse Arena Visualization Library

This script provides a unified entry point for all visualization features
including GUI, command-line interface, and batch processing.
"""

import sys
import argparse
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mouse_arena_visualizer import MouseArenaVisualizer
from plotly_visualizer import AdvancedPlotlyVisualizer
from gui_visualizer import main as gui_main

def launch_gui():
    """Launch the GUI application."""
    print("Launching 3D Mouse Arena Visualizer GUI...")
    gui_main()

def create_basic_plot(mouse_file, arena_file=None, frame_idx=0, output_file=None):
    """Create a basic interactive plot."""
    print(f"Creating basic plot for frame {frame_idx}...")
    
    viz = MouseArenaVisualizer()
    
    # Load mouse data
    if not viz.load_mouse_data(mouse_file):
        print("Error: Could not load mouse data")
        return False
    
    # Load arena data if provided
    if arena_file and viz.load_arena_data(arena_file):
        print("Arena data loaded")
    else:
        print("No arena data provided or could not load")
    
    # Create plot
    fig = viz.create_interactive_plot(
        frame_idx=frame_idx,
        show_arena=(arena_file is not None),
        show_trajectory=False
    )
    
    # Save or show plot
    if output_file:
        fig.write_html(output_file)
        print(f"Plot saved to: {output_file}")
    else:
        fig.show()
    
    return True

def create_animated_plot(mouse_file, arena_file=None, start_frame=0, end_frame=100, 
                       frame_step=5, output_file=None):
    """Create an animated plot."""
    print(f"Creating animated plot from frame {start_frame} to {end_frame}...")
    
    viz = AdvancedPlotlyVisualizer()
    
    # Load data
    if not viz.load_mouse_data(mouse_file):
        print("Error: Could not load mouse data")
        return False
    
    if arena_file and not viz.load_arena_data(arena_file):
        print("Warning: Could not load arena data")
        arena_file = None
    
    # Create animated plot
    fig = viz.create_animated_plot(
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        show_arena=(arena_file is not None),
        show_trajectory=True
    )
    
    # Save or show plot
    if output_file:
        fig.write_html(output_file)
        print(f"Animated plot saved to: {output_file}")
    else:
        fig.show()
    
    return True

def create_multi_view_plot(mouse_file, arena_file=None, frame_idx=0, output_file=None):
    """Create multi-view plot."""
    print(f"Creating multi-view plot for frame {frame_idx}...")
    
    viz = AdvancedPlotlyVisualizer()
    
    # Load data
    if not viz.load_mouse_data(mouse_file):
        print("Error: Could not load mouse data")
        return False
    
    if arena_file and not viz.load_arena_data(arena_file):
        print("Warning: Could not load arena data")
        arena_file = None
    
    # Create multi-view plot
    fig = viz.create_multi_view_plot(
        frame_idx=frame_idx,
        show_arena=(arena_file is not None)
    )
    
    # Save or show plot
    if output_file:
        fig.write_html(output_file)
        print(f"Multi-view plot saved to: {output_file}")
    else:
        fig.show()
    
    return True

def create_heatmap_plot(mouse_file, start_frame=0, end_frame=100, output_file=None):
    """Create heatmap plot."""
    print(f"Creating heatmap plot from frame {start_frame} to {end_frame}...")
    
    viz = AdvancedPlotlyVisualizer()
    
    # Load data
    if not viz.load_mouse_data(mouse_file):
        print("Error: Could not load mouse data")
        return False
    
    # Create heatmap plot
    fig = viz.create_heatmap_plot(
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    if fig is None:
        print("Error: Could not create heatmap plot")
        return False
    
    # Save or show plot
    if output_file:
        fig.write_html(output_file)
        print(f"Heatmap plot saved to: {output_file}")
    else:
        fig.show()
    
    return True

def create_statistics_plot(mouse_file, start_frame=0, end_frame=100, output_file=None):
    """Create statistics plot."""
    print(f"Creating statistics plot from frame {start_frame} to {end_frame}...")
    
    viz = AdvancedPlotlyVisualizer()
    
    # Load data
    if not viz.load_mouse_data(mouse_file):
        print("Error: Could not load mouse data")
        return False
    
    # Create statistics plot
    fig = viz.create_statistics_plot(
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    if fig is None:
        print("Error: Could not create statistics plot")
        return False
    
    # Save or show plot
    if output_file:
        fig.write_html(output_file)
        print(f"Statistics plot saved to: {output_file}")
    else:
        fig.show()
    
    return True

def export_data(mouse_file, arena_file=None, frame_idx=0, output_file=None):
    """Export plot data to JSON."""
    print(f"Exporting data for frame {frame_idx}...")
    
    viz = AdvancedPlotlyVisualizer()
    
    # Load data
    if not viz.load_mouse_data(mouse_file):
        print("Error: Could not load mouse data")
        return False
    
    if arena_file and not viz.load_arena_data(arena_file):
        print("Warning: Could not load arena data")
    
    # Export data
    if output_file is None:
        output_file = f"mouse_arena_data_frame_{frame_idx}.json"
    
    if viz.export_plot_data(output_file, frame_idx):
        print(f"Data exported to: {output_file}")
        return True
    else:
        print("Error: Could not export data")
        return False

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="3D Mouse Arena Visualization Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  python launch_visualizer.py --gui
  
  # Create basic plot
  python launch_visualizer.py --basic mouse_data.h5 --arena arena_data.h5 --frame 0
  
  # Create animated plot
  python launch_visualizer.py --animate mouse_data.h5 --start 0 --end 100 --step 5
  
  # Create multi-view plot
  python launch_visualizer.py --multi mouse_data.h5 --arena arena_data.h5 --frame 50
  
  # Create heatmap
  python launch_visualizer.py --heatmap mouse_data.h5 --start 0 --end 200
  
  # Export data
  python launch_visualizer.py --export mouse_data.h5 --arena arena_data.h5 --frame 0
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--gui', action='store_true', help='Launch GUI application')
    mode_group.add_argument('--basic', metavar='MOUSE_FILE', help='Create basic interactive plot')
    mode_group.add_argument('--animate', metavar='MOUSE_FILE', help='Create animated plot')
    mode_group.add_argument('--multi', metavar='MOUSE_FILE', help='Create multi-view plot')
    mode_group.add_argument('--heatmap', metavar='MOUSE_FILE', help='Create heatmap plot')
    mode_group.add_argument('--stats', metavar='MOUSE_FILE', help='Create statistics plot')
    mode_group.add_argument('--export', metavar='MOUSE_FILE', help='Export data to JSON')
    
    # Common options
    parser.add_argument('--arena', metavar='ARENA_FILE', help='Arena data file (H5 format)')
    parser.add_argument('--frame', type=int, default=0, help='Frame index (default: 0)')
    parser.add_argument('--start', type=int, default=0, help='Start frame for animations/analysis (default: 0)')
    parser.add_argument('--end', type=int, default=100, help='End frame for animations/analysis (default: 100)')
    parser.add_argument('--step', type=int, default=5, help='Frame step for animations (default: 5)')
    parser.add_argument('--output', metavar='OUTPUT_FILE', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        if args.gui:
            launch_gui()
        
        elif args.basic:
            create_basic_plot(
                mouse_file=args.basic,
                arena_file=args.arena,
                frame_idx=args.frame,
                output_file=args.output
            )
        
        elif args.animate:
            create_animated_plot(
                mouse_file=args.animate,
                arena_file=args.arena,
                start_frame=args.start,
                end_frame=args.end,
                frame_step=args.step,
                output_file=args.output
            )
        
        elif args.multi:
            create_multi_view_plot(
                mouse_file=args.multi,
                arena_file=args.arena,
                frame_idx=args.frame,
                output_file=args.output
            )
        
        elif args.heatmap:
            create_heatmap_plot(
                mouse_file=args.heatmap,
                start_frame=args.start,
                end_frame=args.end,
                output_file=args.output
            )
        
        elif args.stats:
            create_statistics_plot(
                mouse_file=args.stats,
                start_frame=args.start,
                end_frame=args.end,
                output_file=args.output
            )
        
        elif args.export:
            export_data(
                mouse_file=args.export,
                arena_file=args.arena,
                frame_idx=args.frame,
                output_file=args.output
            )
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
