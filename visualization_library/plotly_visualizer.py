"""
Advanced Plotly-based 3D Mouse Arena Visualizer

This module provides enhanced Plotly visualization capabilities with
advanced features like animations, multiple views, and interactive controls.
"""

import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import json
from mouse_arena_visualizer import MouseArenaVisualizer

class AdvancedPlotlyVisualizer(MouseArenaVisualizer):
    """
    Advanced Plotly-based visualizer with enhanced features.
    """
    
    def __init__(self):
        super().__init__()
        self.animation_frames = []
        self.camera_positions = {
            'top': dict(eye=dict(x=0, y=0, z=2)),
            'side': dict(eye=dict(x=2, y=0, z=0)),
            'front': dict(eye=dict(x=0, y=2, z=0)),
            'isometric': dict(eye=dict(x=1.5, y=1.5, z=1.5))
        }
    
    def create_animated_plot(self, start_frame=0, end_frame=100, frame_step=1, 
                           individual_idx=0, show_arena=True, show_trajectory=True):
        """Create animated plot showing mouse movement over time."""
        
        if self.mouse_ds is None:
            return None
        
        # Prepare animation frames
        frames = []
        for frame_idx in range(start_frame, min(end_frame, self.mouse_ds.sizes['time']), frame_step):
            frame_data = []
            
            # Add arena if requested
            if show_arena and self.arena_coords is not None:
                arena_mesh = self.create_arena_mesh()
                if arena_mesh:
                    frame_data.append(arena_mesh)
                
                arena_edges = self.create_arena_edges()
                frame_data.extend(arena_edges)
            
            # Add mouse skeleton for current frame
            mouse_traces = self.create_mouse_skeleton(frame_idx, individual_idx)
            frame_data.extend(mouse_traces)
            
            # Add trajectory up to current frame if requested
            if show_trajectory and frame_idx > start_frame:
                trajectory_traces = self.create_trajectory_plot(
                    start_frame, frame_idx, individual_idx
                )
                frame_data.extend(trajectory_traces)
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(frame_idx),
                traces=list(range(len(frame_data)))
            ))
        
        # Create initial plot
        fig = go.Figure()
        
        # Add initial data
        if show_arena and self.arena_coords is not None:
            arena_mesh = self.create_arena_mesh()
            if arena_mesh:
                fig.add_trace(arena_mesh)
            
            arena_edges = self.create_arena_edges()
            for edge in arena_edges:
                fig.add_trace(edge)
        
        # Add initial mouse skeleton
        initial_mouse = self.create_mouse_skeleton(start_frame, individual_idx)
        for trace in initial_mouse:
            fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            title="Animated Mouse Arena Visualization",
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=self.camera_positions['isometric']
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 100}}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}}])
                    ]
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            args=[[f"frame{i}"], {"frame": {"duration": 100}}],
                            label=f"Frame {i}",
                            method="animate"
                        ) for i in range(len(frames))
                    ],
                    active=0,
                    currentvalue={"prefix": "Frame: "}
                )
            ]
        )
        
        # Add frames
        fig.frames = frames
        
        return fig
    
    def create_multi_view_plot(self, frame_idx=0, individual_idx=0, show_arena=True):
        """Create multi-view plot showing different camera angles."""
        
        if self.mouse_ds is None:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=['Top View', 'Side View', 'Front View', 'Isometric View'],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Get mouse coordinates
        coords = self.get_mouse_coordinates(frame_idx, individual_idx)
        if coords is None:
            return None
        
        x, y, z = coords['x'], coords['y'], coords['z']
        names = coords['names']
        
        # Create mouse traces for each view
        for view_idx, (view_name, camera) in enumerate(self.camera_positions.items()):
            row = (view_idx // 2) + 1
            col = (view_idx % 2) + 1
            
            # Add mouse keypoints
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers+text',
                    marker=dict(size=6, color='red'),
                    text=names,
                    textposition="top center",
                    name=f'Mouse {view_name}',
                    showlegend=(view_idx == 0)
                ),
                row=row, col=col
            )
            
            # Add mouse skeleton
            name_to_idx = {name: idx for idx, name in enumerate(names)}
            for start_name, end_name in self.mouse_skeleton:
                if start_name in name_to_idx and end_name in name_to_idx:
                    start_idx = name_to_idx[start_name]
                    end_idx = name_to_idx[end_name]
                    
                    if not (np.isnan(x[start_idx]) or np.isnan(y[start_idx]) or np.isnan(z[start_idx]) or
                            np.isnan(x[end_idx]) or np.isnan(y[end_idx]) or np.isnan(z[end_idx])):
                        
                        fig.add_trace(
                            go.Scatter3d(
                                x=[x[start_idx], x[end_idx]],
                                y=[y[start_idx], y[end_idx]],
                                z=[z[start_idx], z[end_idx]],
                                mode='lines',
                                line=dict(color='blue', width=3),
                                showlegend=False,
                                name='Skeleton'
                            ),
                            row=row, col=col
                        )
            
            # Add arena if requested
            if show_arena and self.arena_coords is not None:
                arena_x, arena_y, arena_z = self.arena_coords['x'], self.arena_coords['y'], self.arena_coords['z']
                
                # Arena points
                fig.add_trace(
                    go.Scatter3d(
                        x=arena_x, y=arena_y, z=arena_z,
                        mode='markers',
                        marker=dict(size=4, color='lightgray'),
                        name='Arena',
                        showlegend=(view_idx == 0)
                    ),
                    row=row, col=col
                )
                
                # Arena edges
                name_to_idx = {name: idx for idx, name in enumerate(self.arena_coords['names'])}
                for start_name, end_name in self.arena_edges:
                    if start_name in name_to_idx and end_name in name_to_idx:
                        start_idx = name_to_idx[start_name]
                        end_idx = name_to_idx[end_name]
                        
                        fig.add_trace(
                            go.Scatter3d(
                                x=[arena_x[start_idx], arena_x[end_idx]],
                                y=[arena_y[start_idx], arena_y[end_idx]],
                                z=[arena_z[start_idx], arena_z[end_idx]],
                                mode='lines',
                                line=dict(color='gray', width=2),
                                showlegend=False,
                                name='Arena Edges'
                            ),
                            row=row, col=col
                        )
            
            # Set camera for this view
            fig.update_scenes(
                camera=camera,
                aspectmode='data',
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'Multi-View Mouse Arena - Frame {frame_idx}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_heatmap_plot(self, start_frame=0, end_frame=100, individual_idx=0, 
                           resolution=10):
        """Create heatmap showing mouse position density."""
        
        if self.mouse_ds is None:
            return None
        
        # Get trajectory data
        mouse_positions = self.mouse_ds.position.isel(individuals=individual_idx).sel(time=slice(start_frame, end_frame))
        
        # Extract all valid positions
        all_x, all_y, all_z = [], [], []
        for kp_name in self.mouse_ds.coords['keypoints'].values:
            x_traj = mouse_positions.sel(keypoints=kp_name, space='x').values
            y_traj = mouse_positions.sel(keypoints=kp_name, space='y').values
            z_traj = mouse_positions.sel(keypoints=kp_name, space='z').values
            
            valid_mask = ~(np.isnan(x_traj) | np.isnan(y_traj) | np.isnan(z_traj))
            all_x.extend(x_traj[valid_mask])
            all_y.extend(y_traj[valid_mask])
            all_z.extend(z_traj[valid_mask])
        
        if not all_x:
            return None
        
        # Create heatmap
        fig = go.Figure()
        
        # Add position density heatmap
        fig.add_trace(go.Histogram2d(
            x=all_x, y=all_y,
            nbinsx=resolution, nbinsy=resolution,
            colorscale='Viridis',
            name='Position Density'
        ))
        
        # Add arena boundaries if available
        if self.arena_coords is not None:
            arena_x, arena_y = self.arena_coords['x'], self.arena_coords['y']
            
            # Create arena boundary
            boundary_x = [arena_x[0], arena_x[1], arena_x[2], arena_x[3], arena_x[0]]
            boundary_y = [arena_y[0], arena_y[1], arena_y[2], arena_y[3], arena_y[0]]
            
            fig.add_trace(go.Scatter(
                x=boundary_x, y=boundary_y,
                mode='lines',
                line=dict(color='red', width=3),
                name='Arena Boundary'
            ))
        
        fig.update_layout(
            title=f'Mouse Position Heatmap (Frames {start_frame}-{end_frame})',
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            showlegend=True
        )
        
        return fig
    
    def create_statistics_plot(self, start_frame=0, end_frame=100, individual_idx=0):
        """Create statistical analysis plot."""
        
        if self.mouse_ds is None:
            return None
        
        # Get trajectory data
        mouse_positions = self.mouse_ds.position.isel(individuals=individual_idx).sel(time=slice(start_frame, end_frame))
        
        # Calculate statistics for each keypoint
        stats_data = []
        for kp_name in self.mouse_ds.coords['keypoints'].values:
            x_traj = mouse_positions.sel(keypoints=kp_name, space='x').values
            y_traj = mouse_positions.sel(keypoints=kp_name, space='y').values
            z_traj = mouse_positions.sel(keypoints=kp_name, space='z').values
            
            valid_mask = ~(np.isnan(x_traj) | np.isnan(y_traj) | np.isnan(z_traj))
            if np.any(valid_mask):
                x_valid, y_valid, z_valid = x_traj[valid_mask], y_traj[valid_mask], z_traj[valid_mask]
                
                # Calculate movement statistics
                distances = np.sqrt(np.diff(x_valid)**2 + np.diff(y_valid)**2 + np.diff(z_valid)**2)
                total_distance = np.sum(distances)
                avg_speed = total_distance / len(distances) if len(distances) > 0 else 0
                
                stats_data.append({
                    'keypoint': kp_name,
                    'total_distance': total_distance,
                    'avg_speed': avg_speed,
                    'x_range': x_valid.max() - x_valid.min(),
                    'y_range': y_valid.max() - y_valid.min(),
                    'z_range': z_valid.max() - z_valid.min(),
                    'x_center': x_valid.mean(),
                    'y_center': y_valid.mean(),
                    'z_center': z_valid.mean()
                })
        
        if not stats_data:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Distance', 'Average Speed', 'X Range', 'Y Range'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        df = pd.DataFrame(stats_data)
        
        # Total distance
        fig.add_trace(
            go.Bar(x=df['keypoint'], y=df['total_distance'], name='Total Distance'),
            row=1, col=1
        )
        
        # Average speed
        fig.add_trace(
            go.Bar(x=df['keypoint'], y=df['avg_speed'], name='Average Speed'),
            row=1, col=2
        )
        
        # X range
        fig.add_trace(
            go.Bar(x=df['keypoint'], y=df['x_range'], name='X Range'),
            row=2, col=1
        )
        
        # Y range
        fig.add_trace(
            go.Bar(x=df['keypoint'], y=df['y_range'], name='Y Range'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Mouse Movement Statistics (Frames {start_frame}-{end_frame})',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def export_plot_data(self, output_path, frame_idx=0, individual_idx=0):
        """Export plot data to JSON format."""
        
        if self.mouse_ds is None:
            return False
        
        try:
            # Get mouse coordinates
            coords = self.get_mouse_coordinates(frame_idx, individual_idx)
            if coords is None:
                return False
            
            # Prepare data
            data = {
                'frame': frame_idx,
                'individual': individual_idx,
                'mouse_keypoints': {
                    'names': coords['names'].tolist(),
                    'x': coords['x'].tolist(),
                    'y': coords['y'].tolist(),
                    'z': coords['z'].tolist()
                },
                'skeleton_connections': self.mouse_skeleton
            }
            
            # Add arena data if available
            if self.arena_coords is not None:
                data['arena'] = {
                    'names': self.arena_coords['names'].tolist(),
                    'x': self.arena_coords['x'].tolist(),
                    'y': self.arena_coords['y'].tolist(),
                    'z': self.arena_coords['z'].tolist()
                }
                data['arena_edges'] = self.arena_edges
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
