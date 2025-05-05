import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import cv2
import xarray as xr
import argparse
from movement.io.load_poses import from_file 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def create_2d_ds(slp_files_dir: Path):
    slp_files = list(slp_files_dir.glob("*.slp"))
    # Windows regex
    cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)predictions\.slp$"



    #mac regex
    #cam_regex = r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([^_]+)_predictions\.slp$" 


    file_path_dict = {re.search(cam_regex, str(f.name)).groups()[0]: f for f in slp_files}
    # From movement.io.load_poses.from_multiview_files, split out here just to fix uppercase inconsistency bug:
    views_list = list(file_path_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")

    dataset_list = [
        from_file(f, source_software="SLEAP")
        for f in file_path_dict.values()
    ]
    # make coordinates labels of the keypoints axis all lowercase
    for ds in dataset_list:
        ds.coords["keypoints"] = ds.coords["keypoints"].str.lower()


    # time_slice = slice(0, 1000)
    ds = xr.concat(dataset_list, dim=new_coord_views)

    bodyparts = list(ds.coords["keypoints"].values)

    print(bodyparts)

    print(ds.position.shape, ds.confidence.shape, bodyparts)

    ds.attrs['fps'] = 'fps'
    ds.attrs['source_file'] = 'sleap'

    return ds

def load_3d_poses(root_dir: Path):
    # Exclude files containing 'reprojection' in the filename
    h5_files = [f for f in root_dir.glob("*.h5") if "reprojection" not in f.name.lower()]
    
    if not h5_files:
        raise FileNotFoundError("No valid .h5 files (excluding reprojection) found in the directory.")
    
    newest_file = max(h5_files, key=lambda f: f.stat().st_mtime)
    poses = xr.open_dataset(newest_file)
    return poses


def load_video_mapping(root_dir, views):
    """Return {view_name: cv2.VideoCapture} by matching filenames."""
    root = Path(root_dir)
    video_files = list(root.glob("*.mp4"))
    mapping = {}
    for view in views:
        for vf in video_files:
            if view in vf.name:
                mapping[view] = cv2.VideoCapture(str(vf))
                break
        if view not in mapping:
            raise FileNotFoundError(f"No video file found for view '{view}'")
    return mapping
def plot_3d_pose_matplotlib(ax, keypoints_3d, keypoint_names, skeleton):
    """
    Plots 3D pose on the given ax with skeleton connections.
    
    Args:
        ax: Matplotlib 3D axis
        keypoints_3d: numpy array (13, 3)
        keypoint_names: list of keypoint labels
        skeleton: list of (start_idx, end_idx) defining bones
    """
    xs, ys, zs = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]
    
    # Plot points
    ax.scatter(xs, ys, zs, c='r', s=30)
    for i, name in enumerate(keypoint_names):
        ax.text(xs[i], ys[i], zs[i], name, fontsize=7)

    # Plot bones
    for start, end in skeleton:
        ax.plot([xs[start], xs[end]], 
                [ys[start], ys[end]], 
                [zs[start], zs[end]], 
                c='gray', linewidth=2)
    
    ax.set_title("3D Pose")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

def get_skeleton(keypoint_names):
    name_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    skeleton_named = [
        ('nose', 'lear'), ('lear', 'rear'), ('rear', 'nose'),
        ('upperback', 'tailbase'), ('upperback', 'uppermid'), ('uppermid', 'upperforward'),
        ('blimbmid', 'rblimb'), ('blimbmid', 'lblimb'),
        ('flimbmid', 'lflimb'), ('flimbmid', 'rflimb'),
        ('upperforward', 'lear'), ('upperforward', 'rear'),
        ('uppermid', 'flimbmid'), ('uppermid', 'blimbmid')
    ]
    return [(name_to_idx[a], name_to_idx[b]) for a, b in skeleton_named]


def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    for x, y in keypoints:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame

def get_skeleton(keypoint_names):
    name_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    skeleton_named = [
        ('nose', 'lear'), ('lear', 'rear'), ('rear', 'nose'),
        ('upperback', 'tailbase'), ('upperback', 'uppermid'), ('uppermid', 'upperforward'),
        ('blimbmid', 'rblimb'), ('blimbmid', 'lblimb'),
        ('flimbmid', 'lflimb'), ('flimbmid', 'rflimb'),
        ('upperforward', 'lear'), ('upperforward', 'rear'),
        ('uppermid', 'flimbmid'), ('uppermid', 'blimbmid')
    ]
    return [(name_to_idx[a], name_to_idx[b]) for a, b in skeleton_named]

def plot_3d_pose_matplotlib(ax, keypoints_3d, keypoint_names, skeleton):
    xs, ys, zs = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]
    ax.scatter(xs, ys, zs, c='r', s=30)
    for i, name in enumerate(keypoint_names):
        ax.text(xs[i], ys[i], zs[i], name, fontsize=7)

    for start, end in skeleton:
        ax.plot([xs[start], xs[end]],
                [ys[start], ys[end]],
                [zs[start], zs[end]],
                c='gray', linewidth=2)

    ax.set_title("3D Pose")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.set_box_aspect([1, 1, 1])

def plot_frame_with_3d(ds_2d, ds_3d, video_map, frame_idx, output_dir=None):
    views = ds_2d.view.values
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 5 views with 2D keypoints
    for i, view in enumerate(views):
        cap = video_map[view]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        kpts = ds_2d['position'].sel(time=frame_idx, view=view).values  # (2, 13, 1)
        kpts = np.squeeze(np.moveaxis(kpts, 0, -1))  # (13, 2)
        frame = draw_keypoints(frame, kpts)

        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(frame)
        ax.set_title(f"{view}")
        ax.axis("off")

    # Plot true 3D keypoints
    ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
    keypoints_3d = ds_3d['position'].sel(time=frame_idx).squeeze().transpose('keypoints', 'space').values  # (13, 3)
    skeleton = get_skeleton(ds_3d.keypoints.values)

    plot_3d_pose_matplotlib(ax_3d, keypoints_3d, list(ds_3d.keypoints.values), skeleton)

    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
    else:
        plt.show()
    plt.close()
def plot_frame_2d_views_only(ds_2d, video_map, frame_idx, output_dir=None):
    views = ds_2d.view.values
    fig = plt.figure(figsize=(20, 6))

    for i, view in enumerate(views):
        cap = video_map[view]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts = ds_2d['position'].sel(time=frame_idx, view=view).values  # (2, 13, 1)
        kpts = np.squeeze(np.moveaxis(kpts, 0, -1))  # (13, 2)
        frame = draw_keypoints(frame, kpts)

        ax = fig.add_subplot(1, 5, i + 1)
        ax.imshow(frame)
        ax.set_title(f"{view}")
        ax.axis("off")

    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--num_frames', type=int)
    args = parser.parse_args()

    root = Path(args.root_dir)
    output_dir = '/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/output/plots'
    num_frames = args.num_frames
    ds = create_2d_ds(root)
    poses3d = load_3d_poses(root)

    print("2D predictions DS:\n", ds.info)
    print("3D poses info \n", poses3d.info)

    video_map = load_video_mapping(root, ds.view.values)

    for frame_idx in range(num_frames):
        print(f"Plotting frame {frame_idx}")
        plot_frame_2d_views_only(ds, video_map, frame_idx, output_dir)

