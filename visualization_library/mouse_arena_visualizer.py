"""
Interactive 3D Mouse Arena Visualizer

This module provides comprehensive visualization tools for mouse tracking data
within experimental arenas using Plotly for interactive 3D plotting.
"""

import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import webbrowser
import tempfile
import os

class MouseArenaVisualizer:
    """
    Interactive 3D visualizer for mouse tracking data within experimental arenas.
    """
    
    def __init__(self):
        self.mouse_ds = None
        self.arena_ds = None
        self.arena_coords = None
        self.mouse_skeleton = self._get_animal_skeleton('mouse')
        self.arena_edges = self._get_arena_edges()
        
    def _get_animal_skeleton(self, animal_type='mouse'):
        """Define animal skeleton connections based on type."""
        if animal_type.lower() == 'mouse':
            return [
                # Head structure
                ('nose', 'ear_lf'), ('nose', 'ear_rt'), ('ear_lf', 'ear_rt'),
                # Spine (backbone)
                ('back_rostral', 'back_mid'), ('back_mid', 'back_caudal'), ('back_caudal', 'tailbase'),
                # Belly (ventral side)
                ('belly_rostral', 'belly_caudal'),
                # Connect head to spine
                ('nose', 'back_rostral'),
                # Connect belly to spine
                ('belly_rostral', 'back_rostral'), ('belly_caudal', 'back_caudal'),
                # Forelimbs (front legs)
                ('forepaw_lf', 'back_rostral'), ('forepaw_rt', 'back_rostral'),
                # Hindlimbs (back legs)
                ('hindpaw_lf', 'back_caudal'), ('hindpaw_rt', 'back_caudal'),
            ]
        elif animal_type.lower() in ['cricket', 'roach', 'insect']:
            # Simplified skeleton for insects (2D flat)
            return [
                # Basic body structure
                ('head', 'thorax'), ('thorax', 'abdomen'),
                # Legs (simplified)
                ('thorax', 'leg_lf'), ('thorax', 'leg_rt'),
                ('abdomen', 'leg_lb'), ('abdomen', 'leg_rb'),
            ]
        else:
            # Generic skeleton
            return [
                ('head', 'body'), ('body', 'tail'),
                ('body', 'leg_lf'), ('body', 'leg_rt'),
            ]
    
    def _get_arena_edges(self):
        """Define arena edge connections for 8-corner rectangular arena."""
        return [
            # Bottom face (floor)
            ('0', '1'), ('1', '2'), ('2', '3'), ('3', '0'),
            # Top face (ceiling)
            ('4', '5'), ('5', '6'), ('6', '7'), ('7', '4'),
            # Vertical edges (walls)
            ('0', '4'), ('1', '5'), ('2', '6'), ('3', '7'),
        ]
    
    def load_mouse_data(self, file_path):
        """Load animal triangulation data from H5 file."""
        try:
            self.mouse_ds = xr.open_dataset(file_path)
            
            # Detect animal type based on individual names and keypoints
            animal_type = self._detect_animal_type()
            self.mouse_skeleton = self._get_animal_skeleton(animal_type)
            self.mouse_ds.attrs['skeleton'] = self.mouse_skeleton
            self.mouse_ds.attrs['animal_type'] = animal_type
            
            print(f"Loaded {animal_type} data: {self.mouse_ds.sizes['time']} frames")
            print(f"Individuals: {list(self.mouse_ds.coords['individuals'].values)}")
            return True
        except Exception as e:
            print(f"Error loading animal data: {e}")
            return False
    
    def _detect_animal_type(self):
        """Detect animal type based on individual names and keypoints."""
        if self.mouse_ds is None:
            return 'unknown'
        
        individuals = list(self.mouse_ds.coords['individuals'].values)
        keypoints = list(self.mouse_ds.coords['keypoints'].values)
        
        # Check individual names
        for individual in individuals:
            individual_lower = individual.lower()
            if 'mouse' in individual_lower or 'rat' in individual_lower:
                return 'mouse'
            elif 'cricket' in individual_lower:
                return 'cricket'
            elif 'roach' in individual_lower or 'cockroach' in individual_lower:
                return 'roach'
            elif 'insect' in individual_lower:
                return 'insect'
        
        # Check keypoint names for mouse-like structure
        mouse_keypoints = ['nose', 'ear_lf', 'ear_rt', 'back_rostral', 'back_mid', 'back_caudal', 'tailbase']
        if any(kp in keypoints for kp in mouse_keypoints):
            return 'mouse'
        
        # Check for insect-like structure
        insect_keypoints = ['head', 'thorax', 'abdomen', 'leg']
        if any(kp in keypoints for kp in insect_keypoints):
            return 'insect'
        
        # Default to mouse if we have typical body parts
        if len(keypoints) > 5:
            return 'mouse'
        
        return 'unknown'
    
    def load_arena_data(self, file_path):
        """Load arena data from H5 file."""
        try:
            self.arena_ds = xr.open_dataset(file_path)
            # Extract arena coordinates
            arena_positions = self.arena_ds.position.isel(time=0, individuals=0)
            self.arena_coords = {
                'x': arena_positions.sel(space='x').values,
                'y': arena_positions.sel(space='y').values,
                'z': arena_positions.sel(space='z').values,
                'names': self.arena_ds.coords['keypoints'].values
            }
            print(f"Loaded arena data: {len(self.arena_coords['names'])} corners")
            return True
        except Exception as e:
            print(f"Error loading arena data: {e}")
            return False
    
    def get_mouse_coordinates(self, frame_idx=0, individual_idx=0):
        """Get mouse coordinates for a specific frame."""
        if self.mouse_ds is None:
            return None
        
        positions = self.mouse_ds.position.isel(time=frame_idx, individuals=individual_idx)
        return {
            'x': positions.sel(space='x').values,
            'y': positions.sel(space='y').values,
            'z': positions.sel(space='z').values,
            'names': self.mouse_ds.coords['keypoints'].values
        }
    
    def create_arena_mesh(self):
        """Create 3D arena mesh from corner points."""
        if self.arena_coords is None:
            return None
        
        # Create arena mesh using the 8 corner points
        x, y, z = self.arena_coords['x'], self.arena_coords['y'], self.arena_coords['z']
        
        # Define faces for the rectangular arena
        # Bottom face (0,1,2,3), Top face (4,5,6,7)
        faces = [
            # Bottom face
            [0, 1, 2], [0, 2, 3],
            # Top face  
            [4, 6, 5], [4, 7, 6],
            # Side faces
            [0, 4, 1], [1, 4, 5],  # Front face
            [1, 5, 2], [2, 5, 6],  # Right face
            [2, 6, 3], [3, 6, 7],  # Back face
            [3, 7, 0], [0, 7, 4],  # Left face
        ]
        
        return go.Mesh3d(
            x=x, y=y, z=z,
            i=[face[0] for face in faces],
            j=[face[1] for face in faces], 
            k=[face[2] for face in faces],
            color='lightgray',
            opacity=0.1,
            name='Arena',
            showlegend=True
        )
    
    def create_arena_edges(self):
        """Create arena edge lines."""
        if self.arena_coords is None:
            return []
        
        edges = []
        x, y, z = self.arena_coords['x'], self.arena_coords['y'], self.arena_coords['z']
        names = self.arena_coords['names']
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        
        for start_name, end_name in self.arena_edges:
            if start_name in name_to_idx and end_name in name_to_idx:
                start_idx = name_to_idx[start_name]
                end_idx = name_to_idx[end_name]
                
                edges.append(go.Scatter3d(
                    x=[x[start_idx], x[end_idx]],
                    y=[y[start_idx], y[end_idx]],
                    z=[z[start_idx], z[end_idx]],
                    mode='lines',
                    line=dict(color='gray', width=4),
                    showlegend=False,
                    name='Arena Edges'
                ))
        
        return edges
    
    def create_mouse_skeleton(self, frame_idx=0, individual_idx=0):
        """Create animal skeleton plot."""
        if self.mouse_ds is None:
            return []
        
        coords = self.get_mouse_coordinates(frame_idx, individual_idx)
        if coords is None:
            return []
        
        traces = []
        x, y, z = coords['x'], coords['y'], coords['z']
        names = coords['names']
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        
        # Get animal type for appropriate coloring
        animal_type = self.mouse_ds.attrs.get('animal_type', 'mouse')
        if animal_type == 'mouse':
            keypoint_color = 'red'
            skeleton_color = 'blue'
            keypoint_size = 10
            skeleton_width = 4
        elif animal_type in ['cricket', 'roach', 'insect']:
            keypoint_color = 'green'
            skeleton_color = 'orange'
            keypoint_size = 8
            skeleton_width = 3
        else:
            keypoint_color = 'purple'
            skeleton_color = 'pink'
            keypoint_size = 8
            skeleton_width = 3
        
        # Plot keypoints with larger, more visible markers
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(size=keypoint_size, color=keypoint_color, line=dict(width=2, color='black')),
            text=names,
            textposition="top center",
            name=f'{animal_type.title()} Keypoints',
            showlegend=True
        ))
        
        # Plot skeleton connections
        for start_name, end_name in self.mouse_skeleton:
            if start_name in name_to_idx and end_name in name_to_idx:
                start_idx = name_to_idx[start_name]
                end_idx = name_to_idx[end_name]
                
                # Check if both points are valid (not NaN)
                if not (np.isnan(x[start_idx]) or np.isnan(y[start_idx]) or np.isnan(z[start_idx]) or
                        np.isnan(x[end_idx]) or np.isnan(y[end_idx]) or np.isnan(z[end_idx])):
                    
                    traces.append(go.Scatter3d(
                        x=[x[start_idx], x[end_idx]],
                        y=[y[start_idx], y[end_idx]],
                        z=[z[start_idx], z[end_idx]],
                        mode='lines',
                        line=dict(color=skeleton_color, width=skeleton_width),
                        showlegend=False,
                        name=f'{animal_type.title()} Skeleton'
                    ))
        
        return traces
    
    def create_2d_flat_visualization(self, frame_idx=0, individual_idx=0):
        """Create 2D flat visualization for insects (cricket/roach)."""
        if self.mouse_ds is None:
            return []
        
        coords = self.get_mouse_coordinates(frame_idx, individual_idx)
        if coords is None:
            return []
        
        traces = []
        x, y, z = coords['x'], coords['y'], coords['z']
        names = coords['names']
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        
        # For insects, project to ground level (z = arena floor)
        if self.arena_coords is not None:
            ground_z = min(self.arena_coords['z'])
            z_flat = np.full_like(z, ground_z)
        else:
            z_flat = np.zeros_like(z)
        
        # Plot keypoints on ground level
        traces.append(go.Scatter3d(
            x=x, y=y, z=z_flat,
            mode='markers+text',
            marker=dict(size=8, color='green'),
            text=names,
            textposition="top center",
            name='Insect Keypoints (2D)',
            showlegend=True
        ))
        
        # Plot skeleton connections on ground level
        for start_name, end_name in self.mouse_skeleton:
            if start_name in name_to_idx and end_name in name_to_idx:
                start_idx = name_to_idx[start_name]
                end_idx = name_to_idx[end_name]
                
                # Check if both points are valid (not NaN)
                if not (np.isnan(x[start_idx]) or np.isnan(y[start_idx]) or np.isnan(z[start_idx]) or
                        np.isnan(x[end_idx]) or np.isnan(y[end_idx]) or np.isnan(z[end_idx])):
                    
                    traces.append(go.Scatter3d(
                        x=[x[start_idx], x[end_idx]],
                        y=[y[start_idx], y[end_idx]],
                        z=[z_flat[start_idx], z_flat[end_idx]],
                        mode='lines',
                        line=dict(color='orange', width=3),
                        showlegend=False,
                        name='Insect Skeleton (2D)'
                    ))
        
        return traces
    
    def create_trajectory_plot(self, start_frame=0, end_frame=100, individual_idx=0):
        """Create trajectory plot for mouse movement."""
        if self.mouse_ds is None:
            return []
        
        traces = []
        mouse_positions = self.mouse_ds.position.isel(individuals=individual_idx).sel(time=slice(start_frame, end_frame))
        keypoint_names = self.mouse_ds.coords['keypoints'].values
        
        # Get arena floor for Z adjustment
        arena_floor = None
        if self.arena_coords is not None:
            arena_floor = min(self.arena_coords['z'])
        
        # Plot trajectory for each keypoint
        colors = px.colors.qualitative.Set3
        for i, kp_name in enumerate(keypoint_names):
            x_traj = mouse_positions.sel(keypoints=kp_name, space='x').values
            y_traj = mouse_positions.sel(keypoints=kp_name, space='y').values
            z_traj = mouse_positions.sel(keypoints=kp_name, space='z').values
            
            # Adjust Z coordinates to arena floor if available
            if arena_floor is not None:
                valid_z = z_traj[~np.isnan(z_traj)]
                if len(valid_z) > 0:
                    mouse_floor = min(valid_z)
                    z_offset = arena_floor - mouse_floor
                    z_traj = z_traj + z_offset
            
            # Only plot valid points
            valid_mask = ~(np.isnan(x_traj) | np.isnan(y_traj) | np.isnan(z_traj))
            if np.any(valid_mask):
                traces.append(go.Scatter3d(
                    x=x_traj[valid_mask],
                    y=y_traj[valid_mask],
                    z=z_traj[valid_mask],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                    name=f'{kp_name} trajectory',
                    opacity=0.6
                ))
        
        return traces
    
    def create_interactive_plot(self, frame_idx=0, show_arena=True, show_trajectory=False, 
                               trajectory_frames=100, individual_idx=0):
        """Create interactive 3D plot with all elements."""
        
        fig = go.Figure()
        
        # Add arena if requested
        if show_arena and self.arena_coords is not None:
            # Add arena mesh
            arena_mesh = self.create_arena_mesh()
            if arena_mesh:
                fig.add_trace(arena_mesh)
            
            # Add arena edges
            arena_edges = self.create_arena_edges()
            for edge in arena_edges:
                fig.add_trace(edge)
        
        # Add animal skeleton based on type
        animal_type = self.mouse_ds.attrs.get('animal_type', 'mouse') if self.mouse_ds else 'mouse'
        
        if animal_type in ['cricket', 'roach', 'insect']:
            # Use 2D flat visualization for insects
            animal_traces = self.create_2d_flat_visualization(frame_idx, individual_idx)
        else:
            # Use 3D visualization for mice and other animals
            animal_traces = self.create_mouse_skeleton(frame_idx, individual_idx)
        
        for trace in animal_traces:
            fig.add_trace(trace)
        
        # Add trajectory if requested
        if show_trajectory:
            trajectory_traces = self.create_trajectory_plot(
                max(0, frame_idx - trajectory_frames), 
                frame_idx, 
                individual_idx
            )
            for trace in trajectory_traces:
                fig.add_trace(trace)
        
        # Update layout for better visualization
        fig.update_layout(
            title=f'Interactive {animal_type.title()} Arena Visualization - Frame {frame_idx}',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=True
        )
        
        return fig
    
    def save_plot(self, fig, output_path):
        """Save plot to HTML file."""
        try:
            fig.write_html(output_path)
            return True
        except Exception as e:
            print(f"Error saving plot: {e}")
            return False
    
    def get_arena_info(self):
        """Get arena boundary information."""
        if self.arena_coords is None:
            return None
        
        x, y, z = self.arena_coords['x'], self.arena_coords['y'], self.arena_coords['z']
        
        return {
            'x_range': (x.min(), x.max()),
            'y_range': (y.min(), y.max()),
            'z_range': (z.min(), z.max()),
            'width': x.max() - x.min(),
            'length': y.max() - y.min(),
            'height': z.max() - z.min(),
            'center': (x.mean(), y.mean(), z.mean())
        }
    
    def get_mouse_info(self, frame_idx=0):
        """Get mouse position information."""
        if self.mouse_ds is None:
            return None
        
        coords = self.get_mouse_coordinates(frame_idx)
        if coords is None:
            return None
        
        x, y, z = coords['x'], coords['y'], coords['z']
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            return None
        
        x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
        
        return {
            'x_range': (x_valid.min(), x_valid.max()),
            'y_range': (y_valid.min(), y_valid.max()),
            'z_range': (z_valid.min(), z_valid.max()),
            'center': (x_valid.mean(), y_valid.mean(), z_valid.mean()),
            'frame': frame_idx,
            'total_frames': self.mouse_ds.sizes['time']
        }
