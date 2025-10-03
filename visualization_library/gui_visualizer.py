"""
GUI Application for Interactive Mouse Arena Visualization

This module provides a tkinter-based GUI for loading and visualizing
mouse tracking data within experimental arenas.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import webbrowser
import tempfile
import os
from pathlib import Path
import xarray as xr
from mouse_arena_visualizer import MouseArenaVisualizer

class MouseArenaGUI:
    """
    GUI application for mouse arena visualization.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("3D Mouse Arena Visualizer")
        self.root.geometry("800x600")
        
        # Initialize visualizer
        self.visualizer = MouseArenaVisualizer()
        
        # GUI variables
        self.mouse_file_path = tk.StringVar()
        self.arena_file_path = tk.StringVar()
        self.frame_var = tk.IntVar(value=0)
        self.show_arena_var = tk.BooleanVar(value=True)
        self.show_trajectory_var = tk.BooleanVar(value=False)
        self.trajectory_frames_var = tk.IntVar(value=100)
        self.individual_var = tk.IntVar(value=0)
        
        # Create GUI
        self.create_widgets()
        
        # Load default arena if available
        self.load_default_arena()
    
    def create_widgets(self):
        """Create GUI widgets."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="3D Mouse Arena Visualizer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Data Files", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Mouse data file
        ttk.Label(file_frame, text="Mouse Data (H5):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.mouse_file_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_mouse_file).grid(row=0, column=2, padx=(0, 0))
        
        # Arena data file
        ttk.Label(file_frame, text="Arena Data (H5):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.arena_file_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_arena_file).grid(row=1, column=2, padx=(0, 0))
        
        # Visualization options
        options_frame = ttk.LabelFrame(main_frame, text="Visualization Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        
        # Frame selection
        ttk.Label(options_frame, text="Frame:").grid(row=0, column=0, sticky=tk.W, pady=2)
        frame_frame = ttk.Frame(options_frame)
        frame_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        self.frame_scale = ttk.Scale(frame_frame, from_=0, to=100, variable=self.frame_var, 
                                   orient=tk.HORIZONTAL, command=self.update_frame_label)
        self.frame_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.frame_label = ttk.Label(frame_frame, text="0")
        self.frame_label.grid(row=0, column=1, padx=(5, 0))
        
        # Individual selection
        ttk.Label(options_frame, text="Individual:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(options_frame, from_=0, to=10, textvariable=self.individual_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Checkboxes
        ttk.Checkbutton(options_frame, text="Show Arena", variable=self.show_arena_var).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Show Trajectory", variable=self.show_trajectory_var).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(20, 0))
        
        # Trajectory frames
        ttk.Label(options_frame, text="Trajectory Frames:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(options_frame, from_=10, to=1000, textvariable=self.trajectory_frames_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=(5, 0))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(button_frame, text="Generate Plot", command=self.generate_plot, 
                  style="Accent.TButton").grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Load Data", command=self.load_data).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Show Info", command=self.show_info).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(button_frame, text="Export Plot", command=self.export_plot).grid(row=0, column=3)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def load_default_arena(self):
        """Load default arena file if available."""
        default_arena = Path("/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/3d-setup/tests/assets/arena_views_triangulated.h5")
        if default_arena.exists():
            self.arena_file_path.set(str(default_arena))
            self.load_arena_data()
    
    def browse_mouse_file(self):
        """Browse for mouse data file."""
        filename = filedialog.askopenfilename(
            title="Select Mouse Data File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if filename:
            self.mouse_file_path.set(filename)
    
    def browse_arena_file(self):
        """Browse for arena data file."""
        filename = filedialog.askopenfilename(
            title="Select Arena Data File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if filename:
            self.arena_file_path.set(filename)
    
    def update_frame_label(self, value):
        """Update frame label when scale changes."""
        self.frame_label.config(text=str(int(float(value))))
    
    def load_data(self):
        """Load mouse and arena data."""
        self.status_var.set("Loading data...")
        self.progress.start()
        
        def load_thread():
            try:
                # Load mouse data
                if self.mouse_file_path.get():
                    if self.visualizer.load_mouse_data(self.mouse_file_path.get()):
                        # Update frame scale
                        max_frames = self.visualizer.mouse_ds.sizes['time'] - 1
                        self.frame_scale.config(to=max_frames)
                        self.frame_var.set(0)
                        self.update_frame_label(0)
                        
                        self.status_var.set(f"Loaded mouse data: {max_frames + 1} frames")
                    else:
                        self.status_var.set("Error loading mouse data")
                        return
                
                # Load arena data
                if self.arena_file_path.get():
                    if self.visualizer.load_arena_data(self.arena_file_path.get()):
                        self.status_var.set("Data loaded successfully")
                    else:
                        self.status_var.set("Error loading arena data")
                        return
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
            finally:
                self.progress.stop()
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def load_arena_data(self):
        """Load arena data only."""
        if self.arena_file_path.get():
            if self.visualizer.load_arena_data(self.arena_file_path.get()):
                self.status_var.set("Arena data loaded")
            else:
                self.status_var.set("Error loading arena data")
    
    def generate_plot(self):
        """Generate interactive plot."""
        if not self.mouse_file_path.get():
            messagebox.showerror("Error", "Please select a mouse data file")
            return
        
        self.status_var.set("Generating plot...")
        self.progress.start()
        
        def plot_thread():
            try:
                # Create plot
                fig = self.visualizer.create_interactive_plot(
                    frame_idx=self.frame_var.get(),
                    show_arena=self.show_arena_var.get(),
                    show_trajectory=self.show_trajectory_var.get(),
                    trajectory_frames=self.trajectory_frames_var.get(),
                    individual_idx=self.individual_var.get()
                )
                
                # Save to temporary file and open
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    temp_path = f.name
                
                fig.write_html(temp_path)
                
                # Open in browser
                webbrowser.open(f'file://{temp_path}')
                
                self.status_var.set("Plot generated and opened in browser")
                
            except Exception as e:
                self.status_var.set(f"Error generating plot: {str(e)}")
                messagebox.showerror("Error", f"Error generating plot: {str(e)}")
            finally:
                self.progress.stop()
        
        threading.Thread(target=plot_thread, daemon=True).start()
    
    def show_info(self):
        """Show data information."""
        info_window = tk.Toplevel(self.root)
        info_window.title("Data Information")
        info_window.geometry("600x400")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(info_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Get information
        info_text = "=== Data Information ===\n\n"
        
        # Mouse data info
        if self.visualizer.mouse_ds is not None:
            mouse_info = self.visualizer.get_mouse_info(self.frame_var.get())
            if mouse_info:
                info_text += f"Mouse Data:\n"
                info_text += f"  Total Frames: {mouse_info['total_frames']}\n"
                info_text += f"  Current Frame: {mouse_info['frame']}\n"
                info_text += f"  Position Range:\n"
                info_text += f"    X: {mouse_info['x_range'][0]:.1f} to {mouse_info['x_range'][1]:.1f} mm\n"
                info_text += f"    Y: {mouse_info['y_range'][0]:.1f} to {mouse_info['y_range'][1]:.1f} mm\n"
                info_text += f"    Z: {mouse_info['z_range'][0]:.1f} to {mouse_info['z_range'][1]:.1f} mm\n"
                info_text += f"  Center: ({mouse_info['center'][0]:.1f}, {mouse_info['center'][1]:.1f}, {mouse_info['center'][2]:.1f}) mm\n\n"
        
        # Arena data info
        if self.visualizer.arena_coords is not None:
            arena_info = self.visualizer.get_arena_info()
            if arena_info:
                info_text += f"Arena Data:\n"
                info_text += f"  Dimensions: {arena_info['width']:.1f} × {arena_info['length']:.1f} × {arena_info['height']:.1f} mm\n"
                info_text += f"  Position Range:\n"
                info_text += f"    X: {arena_info['x_range'][0]:.1f} to {arena_info['x_range'][1]:.1f} mm\n"
                info_text += f"    Y: {arena_info['y_range'][0]:.1f} to {arena_info['y_range'][1]:.1f} mm\n"
                info_text += f"    Z: {arena_info['z_range'][0]:.1f} to {arena_info['z_range'][1]:.1f} mm\n"
                info_text += f"  Center: ({arena_info['center'][0]:.1f}, {arena_info['center'][1]:.1f}, {arena_info['center'][2]:.1f}) mm\n\n"
        
        # Arena coordinates
        if self.visualizer.arena_coords is not None:
            info_text += "Arena Corner Coordinates:\n"
            for i, name in enumerate(self.visualizer.arena_coords['names']):
                x = self.visualizer.arena_coords['x'][i]
                y = self.visualizer.arena_coords['y'][i]
                z = self.visualizer.arena_coords['z'][i]
                info_text += f"  Corner {name}: ({x:8.2f}, {y:8.2f}, {z:8.2f}) mm\n"
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
    
    def export_plot(self):
        """Export plot to file."""
        if not self.mouse_file_path.get():
            messagebox.showerror("Error", "Please select a mouse data file")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Plot",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        
        if filename:
            self.status_var.set("Exporting plot...")
            self.progress.start()
            
            def export_thread():
                try:
                    fig = self.visualizer.create_interactive_plot(
                        frame_idx=self.frame_var.get(),
                        show_arena=self.show_arena_var.get(),
                        show_trajectory=self.show_trajectory_var.get(),
                        trajectory_frames=self.trajectory_frames_var.get(),
                        individual_idx=self.individual_var.get()
                    )
                    
                    fig.write_html(filename)
                    self.status_var.set(f"Plot exported to {filename}")
                    
                except Exception as e:
                    self.status_var.set(f"Error exporting plot: {str(e)}")
                    messagebox.showerror("Error", f"Error exporting plot: {str(e)}")
                finally:
                    self.progress.stop()
            
            threading.Thread(target=export_thread, daemon=True).start()

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = MouseArenaGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
