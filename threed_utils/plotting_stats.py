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


def plot_anchor_distances(
    dataset_original: xr.Dataset,
    dataset_filtered: xr.Dataset,
    keypoint_cols: str | Iterable[str] | None = None,
    max_frame: int = 1000,
    save_path: str | None = None,
    title: str | None = None,
    skeleton: list[tuple[str, str]] | None = None,
):
    """
    Plot anchor distances (distances to connected keypoints) for outlier analysis.
    """
    # Data preparation
    if keypoint_cols is None:
        keypoint_cols = dataset_original.keypoints.values
    elif isinstance(keypoint_cols, str):
        keypoint_cols = [keypoint_cols]
    
    # Select keypoints and slice data
    ds_sel_orig = dataset_original.sel(keypoints=keypoint_cols)
    ds_sel_filt = dataset_filtered.sel(keypoints=keypoint_cols)
    
    if max_frame > 0:
        ds_sel_orig = ds_sel_orig.isel(time=slice(0, max_frame))
        ds_sel_filt = ds_sel_filt.isel(time=slice(0, max_frame))
    
    # Convert to numpy arrays
    arr_orig = ds_sel_orig["position"].values
    arr_filt = ds_sel_filt["position"].values
    
    # Handle array dimensions - convert to (T, S, K) format
    if arr_orig.ndim == 4:
        arr_orig = arr_orig.squeeze(axis=3)
        arr_filt = arr_filt.squeeze(axis=3)
    elif arr_orig.ndim == 3 and arr_orig.shape[1] == 3:
        arr_orig = arr_orig.transpose(1, 2, 0)
        arr_filt = arr_filt.transpose(1, 2, 0)
    
    # Get time coordinate
    t = ds_sel_orig["time"].values
    selected_kps = list(ds_sel_orig.keypoints.values)
    K = max(1, arr_orig.shape[2] if arr_orig.ndim == 3 else 1)
    
    # Create subplots - one row per keypoint
    fig = sp.make_subplots(
        rows=K,
        cols=1,
        subplot_titles=[f"{name} - Distance from Anchors" for name in selected_kps],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add traces for each keypoint
    for i, name in enumerate(selected_kps):
        # Calculate anchor distances
        anchor_dist_orig = None
        anchor_dist_filt = None
        if skeleton is not None:
            connections = [conn for conn in skeleton if name in conn]
            if connections:
                anchor_dists_orig = []
                anchor_dists_filt = []
                for conn in connections:
                    other_kp = conn[1] if conn[0] == name else conn[0]
                    if other_kp in dataset_original.keypoints.values:
                        other_idx_full = list(dataset_original.keypoints.values).index(other_kp)
                        other_pos_orig = dataset_original.position.values[:len(t), :, other_idx_full, 0]
                        other_pos_filt = dataset_filtered.position.values[:len(t), :, other_idx_full, 0]
                        
                        dist_orig = np.linalg.norm(arr_orig[:, :, i].squeeze() - other_pos_orig, axis=1)
                        dist_filt = np.linalg.norm(arr_filt[:, :, i].squeeze() - other_pos_filt, axis=1)
                        anchor_dists_orig.append(dist_orig)
                        anchor_dists_filt.append(dist_filt)
                
                if anchor_dists_orig:
                    anchor_dist_orig = np.mean(anchor_dists_orig, axis=0)
                    anchor_dist_filt = np.mean(anchor_dists_filt, axis=0)
        
        if anchor_dist_orig is not None:
            # Original data
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=anchor_dist_orig,
                    mode='markers',
                    name=f"{name} - Original",
                    marker=dict(color='lightblue', size=3, opacity=0.6),
                    showlegend=True,
                    legendgroup=f"group{i}"
                ),
                row=i+1, col=1
            )
            
            # Filtered data
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=anchor_dist_filt,
                    mode='markers',
                    name=f"{name} - Filtered",
                    marker=dict(color='red', size=3, opacity=0.8),
                    showlegend=True,
                    legendgroup=f"group{i}"
                ),
                row=i+1, col=1
            )
        else:
            # No anchor data - show distance to other selected keypoints instead
            other_kps = [kp for j, kp in enumerate(selected_kps) if j != i]
            if other_kps:
                # Calculate distances to other selected keypoints
                other_dists_orig = []
                other_dists_filt = []
                for other_kp in other_kps:
                    other_idx = selected_kps.index(other_kp)
                    dist_orig = np.linalg.norm(arr_orig[:, :, i].squeeze() - arr_orig[:, :, other_idx].squeeze(), axis=1)
                    dist_filt = np.linalg.norm(arr_filt[:, :, i].squeeze() - arr_filt[:, :, other_idx].squeeze(), axis=1)
                    other_dists_orig.append(dist_orig)
                    other_dists_filt.append(dist_filt)
                
                if other_dists_orig:
                    anchor_dist_orig = np.mean(other_dists_orig, axis=0)
                    anchor_dist_filt = np.mean(other_dists_filt, axis=0)
                    
                    # Original data
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=anchor_dist_orig,
                            mode='markers',
                            name=f"{name} - Original (to other keypoints)",
                            marker=dict(color='lightblue', size=3, opacity=0.6),
                            showlegend=True,
                            legendgroup=f"group{i}"
                        ),
                        row=i+1, col=1
                    )
                    
                    # Filtered data
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=anchor_dist_filt,
                            mode='markers',
                            name=f"{name} - Filtered (to other keypoints)",
                            marker=dict(color='red', size=3, opacity=0.8),
                            showlegend=True,
                            legendgroup=f"group{i}"
                        ),
                        row=i+1, col=1
                    )
                else:
                    # No data at all
                    fig.add_trace(
                        go.Scatter(
                            x=[], y=[],
                            mode='markers',
                            name=f"{name} - No Data",
                            showlegend=False,
                            visible=False
                        ),
                        row=i+1, col=1
                    )
            else:
                # Single keypoint - no anchor data
                fig.add_trace(
                    go.Scatter(
                        x=[], y=[],
                        mode='markers',
                        name=f"{name} - No Anchor Data",
                        showlegend=False,
                        visible=False
                    ),
                    row=i+1, col=1
                )
    
    # Update layout
    if not title:
        title = f"Anchor Distances: Original vs Filtered (n={K})"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        height=max(600, K * 200),
        width=1200,
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Frame", showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Distance from Anchors (units)", showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save if requested
    if save_path is not None:
        fig.write_html(save_path.replace('.png', '.html'))
        print(f"Anchor distances plot saved as: {save_path.replace('.png', '.html')}")
    
    print(f"\n======================================================================")
    print(f"ANCHOR DISTANCES PLOT OPENED IN BROWSER")
    print(f"======================================================================")
    fig.show()
    
    return fig, None


def plot_center_distances(
    dataset_original: xr.Dataset,
    dataset_filtered: xr.Dataset,
    keypoint_cols: str | Iterable[str] | None = None,
    max_frame: int = 1000,
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Plot center distances (distances from mouse center) for outlier analysis.
    """
    # Data preparation
    if keypoint_cols is None:
        keypoint_cols = dataset_original.keypoints.values
    elif isinstance(keypoint_cols, str):
        keypoint_cols = [keypoint_cols]
    
    # Select keypoints and slice data
    ds_sel_orig = dataset_original.sel(keypoints=keypoint_cols)
    ds_sel_filt = dataset_filtered.sel(keypoints=keypoint_cols)
    
    if max_frame > 0:
        ds_sel_orig = ds_sel_orig.isel(time=slice(0, max_frame))
        ds_sel_filt = ds_sel_filt.isel(time=slice(0, max_frame))
    
    # Convert to numpy arrays
    arr_orig = ds_sel_orig["position"].values
    arr_filt = ds_sel_filt["position"].values
    
    # Handle array dimensions - convert to (T, S, K) format
    if arr_orig.ndim == 4:
        arr_orig = arr_orig.squeeze(axis=3)
        arr_filt = arr_filt.squeeze(axis=3)
    elif arr_orig.ndim == 3 and arr_orig.shape[1] == 3:
        arr_orig = arr_orig.transpose(1, 2, 0)
        arr_filt = arr_filt.transpose(1, 2, 0)
    
    # Get time coordinate
    t = ds_sel_orig["time"].values
    selected_kps = list(ds_sel_orig.keypoints.values)
    K = max(1, arr_orig.shape[2] if arr_orig.ndim == 3 else 1)
    
    # Calculate mouse center
    mouse_center_orig = np.mean(arr_orig, axis=(0, 2), keepdims=True)
    mouse_center_filt = np.mean(arr_filt, axis=(0, 2), keepdims=True)
    
    # Create subplots - one row per keypoint
    fig = sp.make_subplots(
        rows=K,
        cols=1,
        subplot_titles=[f"{name} - Distance from Center" for name in selected_kps],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add traces for each keypoint
    for i, name in enumerate(selected_kps):
        # Calculate distance from center
        dist_center_orig = np.linalg.norm(arr_orig[:, :, i].squeeze() - mouse_center_orig[:, :, 0], axis=1)
        dist_center_filt = np.linalg.norm(arr_filt[:, :, i].squeeze() - mouse_center_filt[:, :, 0], axis=1)
        
        # Original data
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dist_center_orig,
                mode='markers',
                name=f"{name} - Original",
                marker=dict(color='lightgreen', size=3, opacity=0.6),
                showlegend=True,
                legendgroup=f"group{i}"
            ),
            row=i+1, col=1
        )
        
        # Filtered data
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dist_center_filt,
                mode='markers',
                name=f"{name} - Filtered",
                marker=dict(color='orange', size=3, opacity=0.8),
                showlegend=True,
                legendgroup=f"group{i}"
            ),
            row=i+1, col=1
        )
    
    # Update layout
    if not title:
        title = f"Center Distances: Original vs Filtered (n={K})"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        height=max(600, K * 200),
        width=1200,
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Frame", showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Distance from Center (units)", showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save if requested
    if save_path is not None:
        fig.write_html(save_path.replace('.png', '.html'))
        print(f"Center distances plot saved as: {save_path.replace('.png', '.html')}")
    
    print(f"\n======================================================================")
    print(f"CENTER DISTANCES PLOT OPENED IN BROWSER")
    print(f"======================================================================")
    fig.show()
    
    return fig, None
