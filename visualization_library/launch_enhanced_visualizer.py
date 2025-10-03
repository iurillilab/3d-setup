#!/usr/bin/env python3
"""
Enhanced Multi-Animal Arena Visualizer Launcher

This script provides both GUI and CLI access to the enhanced multi-animal
visualization system with statistics and advanced features.
"""

import argparse
import sys
import os
from enhanced_gui import run_enhanced_gui
from enhanced_visualizer import EnhancedMultiAnimalVisualizer

def main():
    """Main entry point for the enhanced visualizer."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Animal Arena Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_enhanced_visualizer.py --gui
  python launch_enhanced_visualizer.py --mouse-file data.h5 --arena-file arena.h5
  python launch_enhanced_visualizer.py --mouse-file data.h5 --stats --frames 0 1000
        """
    )
    
    parser.add_argument('--gui', action='store_true',
                       help='Launch the enhanced GUI interface')
    
    parser.add_argument('--mouse-file', type=str,
                       help='Path to mouse/animal data H5 file')
    
    parser.add_argument('--arena-file', type=str,
                       help='Path to arena data H5 file')
    
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame number to visualize (default: 0)')
    
    parser.add_argument('--stats', action='store_true',
                       help='Calculate and display statistics')
    
    parser.add_argument('--frames', nargs=2, type=int, metavar=('START', 'END'),
                       help='Frame range for statistics calculation')
    
    parser.add_argument('--output', type=str,
                       help='Output HTML file path')
    
    parser.add_argument('--show-arena', action='store_true', default=True,
                       help='Show arena in visualization')
    
    parser.add_argument('--show-trajectory', action='store_true',
                       help='Show trajectory in visualization')
    
    parser.add_argument('--trajectory-frames', type=int, default=100,
                       help='Number of frames for trajectory (default: 100)')
    
    args = parser.parse_args()
    
    # Launch GUI if requested
    if args.gui:
        print("Launching Enhanced Multi-Animal Arena Visualizer GUI...")
        run_enhanced_gui()
        return
    
    # CLI mode
    if not args.mouse_file:
        print("Error: --mouse-file is required for CLI mode")
        print("Use --gui for interactive mode")
        sys.exit(1)
    
    print("=== Enhanced Multi-Animal Arena Visualizer ===")
    
    # Initialize visualizer
    visualizer = EnhancedMultiAnimalVisualizer()
    
    # Load mouse data
    print(f"Loading animal data from: {args.mouse_file}")
    if not visualizer.load_multi_animal_data(args.mouse_file):
        print("Error: Failed to load animal data")
        sys.exit(1)
    
    print(f"✓ Loaded {len(visualizer.animals_data)} animals:")
    for name, data in visualizer.animals_data.items():
        print(f"  - {name}: {data['type']}")
    
    # Load arena data if provided
    if args.arena_file:
        print(f"Loading arena data from: {args.arena_file}")
        if visualizer.load_arena_data(args.arena_file):
            print("✓ Arena data loaded")
        else:
            print("⚠ Failed to load arena data")
    
    # Calculate statistics if requested
    if args.stats:
        print("\n=== Calculating Statistics ===")
        start_frame = args.frames[0] if args.frames else 0
        end_frame = args.frames[1] if args.frames else min(1000, visualizer.mouse_ds.sizes['time'])
        
        stats = visualizer.calculate_velocity_statistics(start_frame, end_frame)
        
        if stats:
            print("Velocity Statistics:")
            for animal_name, animal_stats in stats.items():
                print(f"\n{animal_name} ({animal_stats['type']}):")
                print(f"  Mean velocity: {animal_stats['mean_velocity']:.2f} mm/frame")
                print(f"  Max velocity: {animal_stats['max_velocity']:.2f} mm/frame")
                print(f"  Total distance: {animal_stats['total_distance']:.2f} mm")
                print(f"  Frames analyzed: {animal_stats['frames_analyzed']}")
        
        # Calculate distances between animals
        distances = visualizer.calculate_distance_between_animals(args.frame)
        if distances:
            print(f"\nDistances between animals (frame {args.frame}):")
            for pair, dist_info in distances.items():
                print(f"  {pair}: {dist_info['distance_mm']:.2f} mm")
        else:
            print(f"\nOnly one animal detected, no distances to calculate")
    
    # Create visualization
    print(f"\n=== Creating Visualization (Frame {args.frame}) ===")
    try:
        fig = visualizer.create_multi_animal_plot(
            frame_idx=args.frame,
            show_arena=args.show_arena,
            show_trajectory=args.show_trajectory,
            trajectory_frames=args.trajectory_frames
        )
        
        # Save or display plot
        if args.output:
            from plotly.offline import plot
            plot(fig, filename=args.output, auto_open=False)
            print(f"✓ Plot saved to: {args.output}")
        else:
            from plotly.offline import plot
            plot(fig, auto_open=True)
            print("✓ Plot opened in browser")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        sys.exit(1)
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    main()
