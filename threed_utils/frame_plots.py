import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import os
from tqdm import tqdm
import plotly.io as pio

def save_3d_frames_from_pose(ds, skeleton, output_dir, arena_ds=None, num_frames:int=None):
    os.makedirs(output_dir, exist_ok=True)
    keypoint_names = ds.keypoints.values
    keypoint_dict = {name: idx for idx, name in enumerate(keypoint_names)}

    # Convert skeleton names to indices if needed
    if isinstance(skeleton[0][0], str):
        skeleton = [(keypoint_dict[start], keypoint_dict[end]) for start, end in skeleton]

    dataset_color = px.colors.qualitative.Set3[0]
    skeleton_color = 'gray'
    
    n_frames = num_frames if num_frames else ds.sizes['time']
    
    for frame_idx in tqdm(range(n_frames), desc="Saving frames"):
        fig = go.Figure()

        # Arena mesh (optional)
        if arena_ds is not None:
            arena_points = arena_ds.position.isel(time=0, individuals=0).transpose('keypoints', 'space').values
            fig.add_trace(go.Mesh3d(
                x=arena_points[:, 0],
                y=arena_points[:, 1],
                z=arena_points[:, 2],
                color='lightgray',
                opacity=0.3,
                alphahull=0,
                name='arena'
            ))

        # Keypoints
        points = ds.position.isel(time=frame_idx, individuals=0).transpose('keypoints', 'space').values
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color=dataset_color),
            text=[f'{kp}' for kp in keypoint_names],
            hoverinfo='text'
        ))

        # Skeleton
        for start, end in skeleton:
            fig.add_trace(go.Scatter3d(
                x=[points[start, 0], points[end, 0]],
                y=[points[start, 1], points[end, 1]],
                z=[points[start, 2], points[end, 2]],
                mode='lines',
                line=dict(color=skeleton_color, width=2),
                showlegend=False
            ))

        fig.update_layout(scene=dict(aspectmode='data'),
                          margin=dict(l=0, r=0, b=0, t=0),
                          showlegend=False)

        # Save as image
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        pio.write_image(fig, frame_path, width=800, height=600)


pose_ds = xr.open_dataset('/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012/multicam_video_2024-07-22T10_19_22_cropped_20250325101012_triangulated_points_20250327-124608.h5')
arena_ds = xr.open_dataset("/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/3d-setup/tests/assets/arena_views_triangulated.h5")

# Define your skeleton here using keypoint names
skeleton = [
    # Head triangle
    ('nose', 'lear'),
    ('lear', 'rear'),
    ('rear', 'nose'),
    
    # Back connections
    ('upperback', 'tailbase'),
    ('upperback', 'uppermid'),
    ('uppermid', 'upperforward'),
    
    # Back limbs
    ('blimbmid', 'rblimb'),
    ('blimbmid', 'lblimb'),
    
    # Front limbs
    ('flimbmid', 'lflimb'),
    ('flimbmid', 'rflimb'),
    
    # Upper forward connections
    ('upperforward', 'lear'),
    ('upperforward', 'rear'),
    
    # Mid connections
    ('uppermid', 'flimbmid'),
    ('uppermid', 'blimbmid'),
]

save_3d_frames_from_pose(
    ds=pose_ds,
    skeleton=skeleton,
    output_dir="/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data",
    arena_ds=arena_ds,
    num_frames=10
)

# python frame_plots.py \
#     --pose_file /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012/multicam_video_2024-07-22T10_19_22_cropped_20250325101012_triangulated_points_20250327-124608.h5\
#     --output_dir /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/plots \
#     --n_frames 20 \
#     --exclude_kpts flimbmid blimbmid
#     --arena_file  /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/3d-setup/tests/assets/arena_views_triangulated.h5  