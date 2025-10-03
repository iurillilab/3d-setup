"""
Enhanced GUI for Multi-Animal Arena Visualization

This module provides a comprehensive GUI for visualizing multiple animals
with statistics and advanced controls.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import plotly.graph_objects as go
from plotly.offline import plot
import webbrowser
import tempfile
import os
from enhanced_visualizer import EnhancedMultiAnimalVisualizer

class EnhancedVisualizerGUI:
    """Enhanced GUI for multi-animal visualization with statistics."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Multi-Animal Arena Visualizer")
        self.root.geometry("1200x800")
        
        self.visualizer = EnhancedMultiAnimalVisualizer()
        self.current_frame = 0
        self.max_frames = 0
        self.temp_html_file = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Animal Data", 
                  command=self.load_animal_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Load Arena Data", 
                  command=self.load_arena_data).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_status = ttk.Label(file_frame, text="No files loaded")
        self.file_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Visualization Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame control
        frame_control = ttk.Frame(control_frame)
        frame_control.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(frame_control, text="Frame:").pack(side=tk.LEFT)
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(frame_control, from_=0, to=100, 
                                   variable=self.frame_var, orient=tk.HORIZONTAL,
                                   command=self.on_frame_change)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.frame_label = ttk.Label(frame_control, text="0 / 0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Visualization options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_arena_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Arena", 
                       variable=self.show_arena_var).pack(side=tk.LEFT, padx=(0, 10))
        
        self.show_trajectory_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Trajectory", 
                       variable=self.show_trajectory_var).pack(side=tk.LEFT, padx=(0, 10))
        
        self.trajectory_frames_var = tk.IntVar(value=100)
        ttk.Label(options_frame, text="Trajectory frames:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Spinbox(options_frame, from_=10, to=1000, width=10,
                   textvariable=self.trajectory_frames_var).pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Update Plot", 
                  command=self.update_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Animate", 
                  command=self.animate_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Statistics", 
                  command=self.show_statistics).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Open in Browser", 
                  command=self.open_in_browser).pack(side=tk.LEFT)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Statistics text area
        self.stats_text = tk.Text(stats_frame, height=15, wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_animal_data(self):
        """Load animal data file."""
        file_path = filedialog.askopenfilename(
            title="Select Animal Data File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            self.status_var.set("Loading animal data...")
            self.root.update()
            
            if self.visualizer.load_multi_animal_data(file_path):
                self.max_frames = self.visualizer.mouse_ds.sizes['time'] - 1
                self.frame_scale.config(to=self.max_frames)
                self.frame_label.config(text=f"0 / {self.max_frames}")
                
                # Update status
                animals_info = []
                for name, data in self.visualizer.animals_data.items():
                    animals_info.append(f"{name} ({data['type']})")
                
                self.file_status.config(text=f"Loaded: {', '.join(animals_info)}")
                self.status_var.set(f"Loaded {len(self.visualizer.animals_data)} animals")
                
                # Auto-update plot
                self.update_plot()
            else:
                messagebox.showerror("Error", "Failed to load animal data file")
                self.status_var.set("Error loading file")
    
    def load_arena_data(self):
        """Load arena data file."""
        file_path = filedialog.askopenfilename(
            title="Select Arena Data File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            self.status_var.set("Loading arena data...")
            self.root.update()
            
            if self.visualizer.load_arena_data(file_path):
                self.status_var.set("Arena data loaded")
            else:
                messagebox.showerror("Error", "Failed to load arena data file")
                self.status_var.set("Error loading arena file")
    
    def on_frame_change(self, value):
        """Handle frame slider change."""
        self.current_frame = int(float(value))
        self.frame_label.config(text=f"{self.current_frame} / {self.max_frames}")
    
    def update_plot(self):
        """Update the visualization plot."""
        if not self.visualizer.animals_data:
            messagebox.showwarning("Warning", "Please load animal data first")
            return
        
        self.status_var.set("Creating plot...")
        self.root.update()
        
        try:
            fig = self.visualizer.create_multi_animal_plot(
                frame_idx=self.current_frame,
                show_arena=self.show_arena_var.get(),
                show_trajectory=self.show_trajectory_var.get(),
                trajectory_frames=self.trajectory_frames_var.get()
            )
            
            # Save to temporary HTML file
            if self.temp_html_file:
                os.unlink(self.temp_html_file)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                self.temp_html_file = f.name
                plot(fig, filename=self.temp_html_file, auto_open=False)
            
            self.status_var.set("Plot updated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")
            self.status_var.set("Error creating plot")
    
    def animate_plot(self):
        """Create animated plot."""
        if not self.visualizer.animals_data:
            messagebox.showwarning("Warning", "Please load animal data first")
            return
        
        self.status_var.set("Creating animation...")
        self.root.update()
        
        try:
            # Create animation with multiple frames
            frames = []
            step = max(1, self.max_frames // 50)  # Limit to 50 frames for animation
            
            for frame in range(0, self.max_frames, step):
                fig = self.visualizer.create_multi_animal_plot(
                    frame_idx=frame,
                    show_arena=self.show_arena_var.get(),
                    show_trajectory=False  # Don't show trajectory in animation
                )
                frames.append(go.Frame(data=fig.data, name=str(frame)))
            
            # Create animated figure
            animated_fig = go.Figure(
                data=frames[0].data if frames else [],
                frames=frames
            )
            
            # Add animation controls
            animated_fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100}}]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}}]}
                    ]
                }]
            )
            
            # Save animated plot
            if self.temp_html_file:
                os.unlink(self.temp_html_file)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                self.temp_html_file = f.name
                plot(animated_fig, filename=self.temp_html_file, auto_open=False)
            
            self.status_var.set("Animation created successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create animation: {str(e)}")
            self.status_var.set("Error creating animation")
    
    def show_statistics(self):
        """Calculate and display statistics."""
        if not self.visualizer.animals_data:
            messagebox.showwarning("Warning", "Please load animal data first")
            return
        
        self.status_var.set("Calculating statistics...")
        self.root.update()
        
        try:
            # Calculate velocity statistics
            stats = self.visualizer.calculate_velocity_statistics(0, min(1000, self.max_frames))
            
            # Calculate distance between animals
            distances = self.visualizer.calculate_distance_between_animals(self.current_frame)
            
            # Display in text area
            self.stats_text.delete(1.0, tk.END)
            
            self.stats_text.insert(tk.END, "=== ANIMAL STATISTICS ===\n\n")
            
            for animal_name, animal_stats in stats.items():
                self.stats_text.insert(tk.END, f"{animal_name} ({animal_stats['type']}):\n")
                self.stats_text.insert(tk.END, f"  Mean velocity: {animal_stats['mean_velocity']:.2f} mm/frame\n")
                self.stats_text.insert(tk.END, f"  Max velocity: {animal_stats['max_velocity']:.2f} mm/frame\n")
                self.stats_text.insert(tk.END, f"  Total distance: {animal_stats['total_distance']:.2f} mm\n")
                self.stats_text.insert(tk.END, f"  Frames analyzed: {animal_stats['frames_analyzed']}\n\n")
            
            if distances:
                self.stats_text.insert(tk.END, "=== DISTANCES BETWEEN ANIMALS ===\n\n")
                for pair, dist_info in distances.items():
                    self.stats_text.insert(tk.END, f"{pair}: {dist_info['distance_mm']:.2f} mm\n")
            
            self.status_var.set("Statistics calculated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate statistics: {str(e)}")
            self.status_var.set("Error calculating statistics")
    
    def open_in_browser(self):
        """Open the current plot in browser."""
        if self.temp_html_file and os.path.exists(self.temp_html_file):
            webbrowser.open(f"file://{self.temp_html_file}")
        else:
            messagebox.showwarning("Warning", "No plot available. Please create a plot first.")
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()

def run_enhanced_gui():
    """Run the enhanced GUI."""
    app = EnhancedVisualizerGUI()
    app.run()

if __name__ == "__main__":
    run_enhanced_gui()
