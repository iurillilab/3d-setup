import math
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import xarray as xr


def plot_speed_subplots(
    dataset: xr.Dataset,
    keypoint_cols: str | Iterable[str] | None = None,
    max_frame: int = 1000,
    ncols: int = 3,
    figsize: tuple[float, float] = (12, 8),
    sharex: bool = True,
    sharey: bool = False,
    save_path: str | None = None,
    title: str | None = None,
    dataset2: xr.Dataset | None = None,
    dataset2_label: str = "Filtered",
    dataset1_label: str = "Original",
    ylim: tuple[float, float] | None = None,
):
    """
    Make a subplot for each keypoint showing its per-frame speed.

    Speed_k[t] = || position[t+1, :, k] - position[t, :, k] ||_2

    Args:
        dataset: xarray Dataset with 'position' shaped (time, space, keypoints, individuals).
        keypoint_cols: str, iterable of str, or None (plot all keypoints).
        max_frame: slice frames [0:max_frame].
        ncols: number of columns in the subplot grid.
        figsize: overall figure size.
        sharex, sharey: share axes across subplots.
        save_path: if provided, saves the figure.
        dataset2: optional second dataset to compare speeds side by side.
        dataset2_label: label for the second dataset.
        dataset1_label: label for the first dataset.
        ylim: optional tuple (ymin, ymax) to set y-axis limits for zooming.
    Returns:
        (fig, axes)
    """
    assert len(dataset.position.shape) == 4, (
        "Expected (time, space, keypoints, individuals)."
    )

    # --- choose keypoints ---
    if keypoint_cols is None:
        selected_kps = list(dataset.keypoints.values)
    elif isinstance(keypoint_cols, str):
        selected_kps = [keypoint_cols]
    else:
        selected_kps = list(keypoint_cols)

    # --- slice and extract positions for first dataset ---
    fps = 60
    ds_sel = dataset.sel(time=slice(0, max_frame), keypoints=selected_kps)
    pos = ds_sel["position"]
    if "individuals" in pos.dims and pos.sizes.get("individuals", 1) == 1:
        pos = pos.squeeze("individuals")  # (time, space, keypoints)

    arr = pos.values  # (T, S, K) or (K, T, S) if single keypoint
    
    # Handle case where dimensions might be swapped for single keypoint
    if arr.shape[0] == 1 and arr.shape[1] > arr.shape[0]:
        # This is likely (K, T, S) format, transpose to (T, S, K)
        arr = arr.transpose(1, 2, 0)
    
    darr = np.diff(arr, axis=0)  # (T-1, S, K)
    speed1 = np.linalg.norm(darr, axis=1)  # (T-1, K) or (T-1,) if K==1

    # time coordinate for the diff'ed series
    t = ds_sel["time"].values
    t = t[1:] if t.size > 1 else t

    # ensure 2D for uniform plotting
    if speed1.ndim == 1:
        speed1 = speed1[:, None]

    # --- process second dataset if provided ---
    speed2 = None
    t2 = None
    if dataset2 is not None:
        assert len(dataset2.position.shape) == 4, (
            "Expected (time, space, keypoints, individuals) for second dataset."
        )
        ds_sel2 = dataset2.sel(time=slice(0, max_frame), keypoints=selected_kps)
        pos2 = ds_sel2["position"]
        if "individuals" in pos2.dims and pos2.sizes.get("individuals", 1) == 1:
            pos2 = pos2.squeeze("individuals")  # (time, space, keypoints)

        arr2 = pos2.values  # (T, S, K) or (K, T, S) if single keypoint
        
        # Handle case where dimensions might be swapped for single keypoint
        if arr2.shape[0] == 1 and arr2.shape[1] > arr2.shape[0]:
            # This is likely (K, T, S) format, transpose to (T, S, K)
            arr2 = arr2.transpose(1, 2, 0)
        
        darr2 = np.diff(arr2, axis=0)  # (T-1, S, K)
        speed2 = np.linalg.norm(darr2, axis=1)  # (T-1, K) or (T-1,) if K==1
        
        # time coordinate for the second dataset
        t2 = ds_sel2["time"].values
        t2 = t2[1:] if t2.size > 1 else t2
        
        # ensure 2D for uniform plotting
        if speed2.ndim == 1:
            speed2 = speed2[:, None]

    K = speed1.shape[1]
    
    # Set reasonable y-axis limits if not provided
    if ylim is None:
        # Calculate reasonable y-axis limits based on the data
        all_speeds = []
        if dataset2 is not None:
            all_speeds.extend([speed1.flatten(), speed2.flatten()])
        else:
            all_speeds.append(speed1.flatten())
        
        all_speeds_flat = np.concatenate(all_speeds)
        # Use 95th percentile to avoid extreme outliers
        y_max = np.percentile(all_speeds_flat, 95)
        ylim = (0, y_max)
        print(f"Auto-setting y-axis limit to: {ylim[1]:.1f} (95th percentile)")
    
    # Create Plotly subplots with better layout
    if dataset2 is not None:
        # For comparison plots, use a single column with larger subplots
        nrows = K
        ncols = 1
        
        # Create subplot titles
        subplot_titles = []
        for name in selected_kps:
            subplot_titles.append(f"{name} - Speed Comparison")
        
        fig = sp.make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            shared_xaxes=True,  # Share x-axis for easy comparison
            vertical_spacing=0.15,  # More space between subplots
            horizontal_spacing=0.1
        )
        
        # Add traces for each keypoint - both datasets on same subplot
        for i, name in enumerate(selected_kps):
            # Add original data trace
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=speed1[:, i],
                    mode='lines',
                    name=f"{name} - {dataset1_label}",
                    line=dict(color='blue', width=2),
                    showlegend=True,
                    legendgroup=f"group{i}"
                ),
                row=i+1, col=1
            )
            
            # Add filtered data trace
            fig.add_trace(
                go.Scatter(
                    x=t2,
                    y=speed2[:, i],
                    mode='lines',
                    name=f"{name} - {dataset2_label}",
                    line=dict(color='orange', width=2),
                    showlegend=True,
                    legendgroup=f"group{i}"
                ),
                row=i+1, col=1
            )
    else:
        # Single dataset plotting
        nrows = math.ceil(K / ncols)
        
        fig = sp.make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=selected_kps,
            shared_yaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add traces for each keypoint
        for i, name in enumerate(selected_kps):
            r, c = divmod(i, ncols)
            
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=speed1[:, i],
                    mode='lines',
                    name=name,
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=r+1, col=c+1
            )
    
    # Update layout
    if not title:
        if dataset2 is not None:
            title = f"Keypoint Speed Comparison: {dataset1_label} vs {dataset2_label} (n={K})"
        else:
            title = f"Keypoint Speeds (n={K})"
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial Black"),
            x=0.5  # Center the title
        ),
        height=max(800, K * 300),  # Dynamic height based on number of keypoints
        width=1200,  # Fixed width for better readability
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        margin=dict(l=60, r=60, t=100, b=60)  # Better margins
    )
    
    # Update axes
    fig.update_xaxes(title_text="Frame", showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Speed (units/frame)", showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Set y-axis limits
    fig.update_yaxes(range=ylim)
    
    # Save if requested
    if save_path is not None:
        fig.write_html(save_path.replace('.png', '.html'))
        print(f"Plot saved as: {save_path.replace('.png', '.html')}")
    
    # Add helpful instructions
    print("\n" + "="*70)
    print("PLOTLY INTERACTIVE PLOT OPENED IN BROWSER")
    print("="*70)
    print("INTERACTIVE FEATURES:")
    print("• Mouse wheel: Zoom in/out")
    print("• Click & drag: Pan around")
    print("• Double-click: Reset zoom")
    print("• Hover: See exact values")
    print("• Legend: Click to show/hide traces")
    print("• Toolbar: Use zoom, pan, and download buttons")
    print("• Box zoom: Draw rectangle to zoom to that area")
    print("")
    print("COMPARISON FEATURES:")
    print("• Blue lines: Original data")
    print("• Orange lines: Filtered data")
    print("• Each keypoint has its own subplot")
    print("• All subplots share the same x-axis for easy comparison")
    print("="*70)
    
    # Show the plot
    fig.show()
    
    return fig, None  # Return None for axes to maintain compatibility
