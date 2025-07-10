import os
import argparse
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr

def plot_frame(datasets, frame_idx, arena_ds=None, labels=None,
               skeleton=None, output_dir='frames', exclude_keypoints=None):
    if not isinstance(datasets, list):
        datasets = [datasets]
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]

    fig_data = []
    colors = px.colors.qualitative.Set3
    skeleton_colors = ['rgba(100,100,100,0.8)', 'rgba(150,150,150,0.8)', 'rgba(200,200,200,0.8)']

    # Arena
    if arena_ds is not None:
        arena_points = arena_ds.position.isel(time=0, individuals=0).transpose('keypoints', 'space').values
        fig_data.append(go.Mesh3d(
            x=arena_points[:, 0],
            y=arena_points[:, 1],
            z=arena_points[:, 2],
            color='lightgrey',
            opacity=0.2,
            alphahull=0,
            name='arena'
        ))

    for idx, (ds, label) in enumerate(zip(datasets, labels)):
        keypoints = ds.keypoints.values.tolist()
        kp_index = {k: i for i, k in enumerate(keypoints)}
        points = ds.position.isel(time=frame_idx, individuals=0).transpose('keypoints', 'space').values

        # Apply exclusion
        if exclude_keypoints:
            include_idxs = [i for i, name in enumerate(keypoints) if name not in exclude_keypoints]
        else:
            include_idxs = list(range(len(keypoints)))

        fig_data.append(go.Scatter3d(
            x=points[include_idxs, 0],
            y=points[include_idxs, 1],
            z=points[include_idxs, 2],
            mode='markers',
            marker=dict(size=5, color=colors[idx]),
            text=[keypoints[i] for i in include_idxs],
            name=label
        ))

        if skeleton:
            for start, end in skeleton:
                if (start not in kp_index or end not in kp_index):
                    continue
                if exclude_keypoints and (start in exclude_keypoints or end in exclude_keypoints):
                    continue
                i, j = kp_index[start], kp_index[end]
                fig_data.append(go.Scatter3d(
                    x=[points[i, 0], points[j, 0]],
                    y=[points[i, 1], points[j, 1]],
                    z=[points[i, 2], points[j, 2]],
                    mode='lines',
                    line=dict(width=2, color=skeleton_colors[idx % len(skeleton_colors)]),
                    showlegend=False
                ))

    fig = go.Figure(data=fig_data)
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
    fig.write_image(out_path, scale=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_files', nargs='+', required=True)
    parser.add_argument('--arena_file', type=str, required=False)
    parser.add_argument('--output_dir', type=str, default='frames_output')
    parser.add_argument('--pose_labels', nargs='+')
    parser.add_argument('--exclude_keypoints', type=str, help='Comma-separated list to exclude', default=None)
    parser.add_argument('--time_slice', nargs=2, type=int, help='Start and end frame indices', default=None)

    args = parser.parse_args()

    datasets = [xr.open_dataset(p) for p in args.pose_files]
    if args.time_slice:
        start, end = args.time_slice
        datasets = [ds.sel(time=slice(start, end)) for ds in datasets]

    arena_ds = xr.open_dataset(args.arena_file) if args.arena_file else None
    labels = args.pose_labels or [f"Pose {i+1}" for i in range(len(datasets))]
    exclude = args.exclude_keypoints.split(',') if args.exclude_keypoints else None
    num_frames = datasets[0].sizes['time']

    skeleton = [
        ('nose', 'lear'), ('lear', 'rear'), ('rear', 'nose'),
        ('upperback', 'tailbase'), ('upperback', 'uppermid'),
        ('uppermid', 'upperforward'),
        ('blimbmid', 'rblimb'), ('blimbmid', 'lblimb'),
        ('flimbmid', 'lflimb'), ('flimbmid', 'rflimb'),
        ('upperforward', 'lear'), ('upperforward', 'rear'),
        ('uppermid', 'flimbmid'), ('uppermid', 'blimbmid')
    ]

    print(f"Rendering {num_frames} frames to: {args.output_dir}")
    for i in tqdm(range(num_frames)):
        plot_frame(datasets, i, arena_ds, labels, skeleton, args.output_dir, exclude)

if __name__ == "__main__":
    main()

# python animation_tools.py --pose_files  /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012/multicam_video_2024-07-22T10_19_22_cropped_20250325101012_triangulated_points_20250327-124608.h5 \
#           --arena_file /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/newarena.h5\
#              --output /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/animation_try_filtered.html \
#             --speed 200 \
#             --pose_labels  "anipose_optimised extra"
# python render_poses.py \
#     --pose_files /Users/thomasbush/Downloads/multicam_video_2024-08-03T11_19_55_cropped_20250325101012/multicam_video_2024-08-03T11_19_55_cropped_20250325101012_triangulated_points_20250401-145955.h5 \
#     --arena_file /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/newarena.h5 \
#     --pose_labels "Mouse 1" "Mouse 2" \
#     --output_dir  /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/plots/plots1 \
#     --exclude_keypoints flimbmid,blimbmid \
#     --time_slice 0 300