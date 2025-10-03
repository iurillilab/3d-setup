"""
Enhanced Multi-Animal Arena Visualizer

This module provides advanced visualization capabilities for multiple animals
within experimental arenas, including statistics and proper arena positioning.
"""

import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from mouse_arena_visualizer import MouseArenaVisualizer

class EnhancedMultiAnimalVisualizer(MouseArenaVisualizer):
    """
    Enhanced visualizer with multi-animal support and statistics.
    """
    
    def __init__(self):
        super().__init__()
        self.animals_data = {}
        self.animal_types = {}
        self.statistics = {}
    
    def load_multi_animal_data(self, file_path):
        """Load data and detect all animals present."""
        try:
            self.mouse_ds = xr.open_dataset(file_path)
            
            # Detect all animals in the dataset
            individuals = list(self.mouse_ds.coords['individuals'].values)
            keypoints = list(self.mouse_ds.coords['keypoints'].values)
            
            print(f"Found {len(individuals)} individuals: {individuals}")
            print(f"Keypoints: {keypoints}")
            
            # Analyze each individual
            for i, individual in enumerate(individuals):
                animal_type = self._detect_animal_type_from_data(individual, keypoints, i)
                self.animal_types[individual] = animal_type
                
                # Get coordinates for this individual
                coords = self._get_individual_coordinates(i)
                self.animals_data[individual] = {
                    'type': animal_type,
                    'coordinates': coords,
                    'index': i
                }
                
                print(f"Individual '{individual}': {animal_type}")
            
            # Set skeleton based on detected animals
            self._set_appropriate_skeleton()
            
            return True
            
        except Exception as e:
            print(f"Error loading multi-animal data: {e}")
            return False
    
    def _detect_animal_type_from_data(self, individual_name, keypoints, individual_idx):
        """Detect animal type from individual name and keypoints."""
        individual_lower = individual_name.lower()
        
        # Check individual name
        if 'mouse' in individual_lower or 'rat' in individual_lower:
            return 'mouse'
        elif 'cricket' in individual_lower:
            return 'cricket'
        elif 'roach' in individual_lower or 'cockroach' in individual_lower:
            return 'roach'
        elif 'insect' in individual_lower:
            return 'insect'
        
        # Check keypoints for mouse-like structure
        mouse_keypoints = ['nose', 'ear_lf', 'ear_rt', 'back_rostral', 'back_mid', 'back_caudal', 'tailbase']
        if any(kp in keypoints for kp in mouse_keypoints):
            return 'mouse'
        
        # Check for insect-like structure
        insect_keypoints = ['head', 'thorax', 'abdomen', 'leg']
        if any(kp in keypoints for kp in insect_keypoints):
            return 'insect'
        
        # Check coordinate ranges to distinguish mouse from arena
        if self.mouse_ds is not None:
            pos = self.mouse_ds.position.isel(individuals=individual_idx, time=0)
            x = pos.sel(space='x').values
            y = pos.sel(space='y').values
            z = pos.sel(space='z').values
            
            # Mouse data typically has smaller coordinate ranges
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            z_range = z.max() - z.min()
            
            if x_range < 100 and y_range < 100 and z_range < 50:
                return 'mouse'
            elif x_range > 200 or y_range > 200:
                return 'arena'
        
        return 'unknown'
    
    def _get_individual_coordinates(self, individual_idx):
        """Get coordinates for a specific individual."""
        positions = self.mouse_ds.position.isel(individuals=individual_idx)
        return {
            'x': positions.sel(space='x').values,
            'y': positions.sel(space='y').values,
            'z': positions.sel(space='z').values,
            'names': self.mouse_ds.coords['keypoints'].values
        }
    
    def _set_appropriate_skeleton(self):
        """Set skeleton based on detected animals."""
        # Use the first animal's skeleton as default
        if self.animals_data:
            first_animal = list(self.animals_data.values())[0]
            animal_type = first_animal['type']
            self.mouse_skeleton = self._get_animal_skeleton(animal_type)
    
    def create_multi_animal_plot(self, frame_idx=0, show_arena=True, show_trajectory=False, 
                               trajectory_frames=100):
        """Create plot showing all animals in the arena."""
        
        fig = go.Figure()
        
        # Add arena if requested
        if show_arena and self.arena_coords is not None:
            arena_mesh = self.create_arena_mesh()
            if arena_mesh:
                fig.add_trace(arena_mesh)
            
            arena_edges = self.create_arena_edges()
            for edge in arena_edges:
                fig.add_trace(edge)
        
        # Add each animal
        colors = px.colors.qualitative.Set3
        for i, (animal_name, animal_data) in enumerate(self.animals_data.items()):
            animal_type = animal_data['type']
            individual_idx = animal_data['index']
            
            if animal_type in ['cricket', 'roach', 'insect']:
                # Use 2D flat visualization for insects
                animal_traces = self.create_2d_flat_visualization(frame_idx, individual_idx)
            else:
                # Use 3D visualization for mice and other animals
                animal_traces = self.create_animal_skeleton(frame_idx, individual_idx, animal_type, colors[i % len(colors)])
            
            for trace in animal_traces:
                trace.name = f'{animal_name} ({animal_type})'
                fig.add_trace(trace)
        
        # Add trajectory if requested
        if show_trajectory:
            for animal_name, animal_data in self.animals_data.items():
                individual_idx = animal_data['index']
                trajectory_traces = self.create_trajectory_plot(
                    max(0, frame_idx - trajectory_frames), 
                    frame_idx, 
                    individual_idx
                )
                for trace in trajectory_traces:
                    trace.name = f'{animal_name} trajectory'
                    fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            title=f'Multi-Animal Arena Visualization - Frame {frame_idx}',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=True
        )
        
        return fig
    
    def create_animal_skeleton(self, frame_idx=0, individual_idx=0, animal_type='mouse', color='red'):
        """Create skeleton for a specific animal type."""
        if self.mouse_ds is None:
            return []
        
        # Get coordinates for this individual
        positions = self.mouse_ds.position.isel(time=frame_idx, individuals=individual_idx)
        x = positions.sel(space='x').values
        y = positions.sel(space='y').values
        z = positions.sel(space='z').values
        names = self.mouse_ds.coords['keypoints'].values
        
        # Get appropriate skeleton for this animal type
        skeleton = self._get_animal_skeleton(animal_type)
        
        traces = []
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        
        # Plot keypoints
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(size=10, color=color, line=dict(width=2, color='black')),
            text=names,
            textposition="top center",
            name=f'{animal_type.title()} Keypoints',
            showlegend=True
        ))
        
        # Plot skeleton connections
        for start_name, end_name in skeleton:
            if start_name in name_to_idx and end_name in name_to_idx:
                start_idx = name_to_idx[start_name]
                end_idx = name_to_idx[end_name]
                
                if not (np.isnan(x[start_idx]) or np.isnan(y[start_idx]) or np.isnan(z[start_idx]) or
                        np.isnan(x[end_idx]) or np.isnan(y[end_idx]) or np.isnan(z[end_idx])):
                    
                    traces.append(go.Scatter3d(
                        x=[x[start_idx], x[end_idx]],
                        y=[y[start_idx], y[end_idx]],
                        z=[z[start_idx], z[end_idx]],
                        mode='lines',
                        line=dict(color=color, width=4),
                        showlegend=False,
                        name=f'{animal_type.title()} Skeleton'
                    ))
        
        return traces
    
    def calculate_velocity_statistics(self, start_frame=0, end_frame=None):
        """Calculate velocity statistics for all animals."""
        if self.mouse_ds is None:
            return {}
        
        if end_frame is None:
            end_frame = self.mouse_ds.sizes['time']
        
        statistics = {}
        
        for animal_name, animal_data in self.animals_data.items():
            individual_idx = animal_data['index']
            animal_type = animal_data['type']
            
            # Get position data for this animal
            positions = self.mouse_ds.position.isel(individuals=individual_idx).sel(time=slice(start_frame, end_frame))
            
            # Calculate velocity for each keypoint
            velocities = []
            for kp_name in self.mouse_ds.coords['keypoints'].values:
                x_traj = positions.sel(keypoints=kp_name, space='x').values
                y_traj = positions.sel(keypoints=kp_name, space='y').values
                z_traj = positions.sel(keypoints=kp_name, space='z').values
                
                # Calculate velocity (distance between consecutive frames)
                valid_mask = ~(np.isnan(x_traj) | np.isnan(y_traj) | np.isnan(z_traj))
                if np.any(valid_mask):
                    x_valid = x_traj[valid_mask]
                    y_valid = y_traj[valid_mask]
                    z_valid = z_traj[valid_mask]
                    
                    # Calculate distances between consecutive points
                    distances = np.sqrt(np.diff(x_valid)**2 + np.diff(y_valid)**2 + np.diff(z_valid)**2)
                    velocities.extend(distances)
            
            if velocities:
                statistics[animal_name] = {
                    'type': animal_type,
                    'mean_velocity': np.mean(velocities),
                    'max_velocity': np.max(velocities),
                    'std_velocity': np.std(velocities),
                    'total_distance': np.sum(velocities),
                    'frames_analyzed': end_frame - start_frame
                }
        
        self.statistics = statistics
        return statistics
    
    def calculate_distance_between_animals(self, frame_idx=0):
        """Calculate distance between animals at a specific frame."""
        if len(self.animals_data) < 2:
            return {}
        
        distances = {}
        animal_names = list(self.animals_data.keys())
        
        for i in range(len(animal_names)):
            for j in range(i + 1, len(animal_names)):
                animal1 = animal_names[i]
                animal2 = animal_names[j]
                
                # Get center positions for both animals
                pos1 = self.mouse_ds.position.isel(time=frame_idx, individuals=self.animals_data[animal1]['index'])
                pos2 = self.mouse_ds.position.isel(time=frame_idx, individuals=self.animals_data[animal2]['index'])
                
                # Calculate center of mass for each animal
                x1 = np.nanmean(pos1.sel(space='x').values)
                y1 = np.nanmean(pos1.sel(space='y').values)
                z1 = np.nanmean(pos1.sel(space='z').values)
                
                x2 = np.nanmean(pos2.sel(space='x').values)
                y2 = np.nanmean(pos2.sel(space='y').values)
                z2 = np.nanmean(pos2.sel(space='z').values)
                
                # Calculate 3D distance
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                distances[f'{animal1}_to_{animal2}'] = {
                    'distance_mm': distance,
                    'animal1_type': self.animals_data[animal1]['type'],
                    'animal2_type': self.animals_data[animal2]['type']
                }
        
        return distances
    
    def create_statistics_plot(self, start_frame=0, end_frame=100):
        """Create statistics visualization."""
        if not self.statistics:
            self.calculate_velocity_statistics(start_frame, end_frame)
        
        if not self.statistics:
            return None
        
        # Create subplots for different statistics
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean Velocity', 'Max Velocity', 'Total Distance', 'Animal Types'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        animals = list(self.statistics.keys())
        mean_velocities = [self.statistics[animal]['mean_velocity'] for animal in animals]
        max_velocities = [self.statistics[animal]['max_velocity'] for animal in animals]
        total_distances = [self.statistics[animal]['total_distance'] for animal in animals]
        animal_types = [self.statistics[animal]['type'] for animal in animals]
        
        # Mean velocity
        fig.add_trace(
            go.Bar(x=animals, y=mean_velocities, name='Mean Velocity'),
            row=1, col=1
        )
        
        # Max velocity
        fig.add_trace(
            go.Bar(x=animals, y=max_velocities, name='Max Velocity'),
            row=1, col=2
        )
        
        # Total distance
        fig.add_trace(
            go.Bar(x=animals, y=total_distances, name='Total Distance'),
            row=2, col=1
        )
        
        # Animal types pie chart
        type_counts = pd.Series(animal_types).value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values, name='Animal Types'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Animal Statistics (Frames {start_frame}-{end_frame})',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def print_statistics_summary(self):
        """Print a summary of calculated statistics."""
        if not self.statistics:
            print("No statistics calculated yet. Run calculate_velocity_statistics() first.")
            return
        
        print("=== ANIMAL STATISTICS SUMMARY ===")
        for animal_name, stats in self.statistics.items():
            print(f"\n{animal_name} ({stats['type']}):")
            print(f"  Mean velocity: {stats['mean_velocity']:.2f} mm/frame")
            print(f"  Max velocity: {stats['max_velocity']:.2f} mm/frame")
            print(f"  Total distance: {stats['total_distance']:.2f} mm")
            print(f"  Frames analyzed: {stats['frames_analyzed']}")
        
        # Distance between animals
        distances = self.calculate_distance_between_animals(0)
        if distances:
            print(f"\n=== DISTANCES BETWEEN ANIMALS (Frame 0) ===")
            for pair, dist_info in distances.items():
                print(f"{pair}: {dist_info['distance_mm']:.2f} mm")
