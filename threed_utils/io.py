import numpy as np
from movement.io.load_poses import from_file, from_multiview_files, from_numpy
import toml
import xarray as xr
from datetime import datetime
from pathlib import Path
from pprint import pprint

SKELETON = [('nose', 'ear_lf'), 
                            ('nose', 'ear_rt'),
                           ('ear_lf', 'ear_rt'),
                           ('ear_lf', 'back_rostral'),
                           ('ear_rt', 'back_rostral'),
                           ('back_rostral', 'back_mid'),
                           ('back_mid', 'back_caudal'),
                           ('back_caudal', 'tailbase'),
                           ('tailbase', 'belly_caudal'),
                           ('belly_caudal', 'belly_rostral'),
                           ('back_caudal', 'hindpaw_lf'),
                           ('back_caudal', 'hindpaw_rt'),
                           ('back_mid', 'hindpaw_lf'),
                           ('back_mid', 'hindpaw_rt'),
                           ('back_mid', 'forepaw_lf'),
                           ('back_mid', 'forepaw_rt'),
                           ('back_rostral', 'forepaw_lf'),
                           ('back_rostral', 'forepaw_rt')]

def movement_ds_from_anipose_triangulation_df(triang_df, individual_name="checkerboard"):
    """Convert triangulation dataframe to xarray dataset.
    Reshape dataframe with columns keypoint1_x, keypoint1_y, keypoint1_z, keypoint1_confidence_score, 
    keypoint2_x, keypoint2_y, keypoint2_z, keypoint2_confidence_score, ...
    to array of positions with dimensions time, individuals, keypoints, space,
    and array of confidence scores with dimensions time, individuals, keypoints
    """
    keypoint_names = sorted(list(set([col.rsplit('_', 1)[0] for col in triang_df.columns 
                                   if any(col.endswith(f'_{s}') for s in ['x','y','z'])])))

    n_frames = len(triang_df)
    n_keypoints = len(keypoint_names)


    # Initialize arrays and fill
    position_array = np.zeros((n_frames, 1, n_keypoints, 3))  # 1 for single individual
    print(position_array.shape)
    confidence_array = np.zeros((n_frames, 1, n_keypoints))
    print(confidence_array.shape)
    for i, kp in enumerate(keypoint_names):
        for j, coord in enumerate(['x', 'y', 'z']):
            position_array[:, 0, i, j] = triang_df[f'{kp}_{coord}']
        confidence_array[:, 0, i] = triang_df[f'{kp}_score']

    individual_names = [individual_name]
    position_array = position_array.transpose(0, 3, 2, 1)
    confidence_array = confidence_array.transpose(0, 2, 1)
    print("array before numpy")
    print(position_array.shape)
    print(confidence_array.shape)

    return from_numpy(position_array=position_array,
                     confidence_array=confidence_array, 
                     individual_names=individual_names,
                     keypoint_names=keypoint_names,
                     source_software="anipose_triangulation")


def write_calibration_toml(output_path, cam_names, img_sizes, extrinsics, intrinsics, result):
    """Write calibration data to TOML format"""
    calibration_dict = dict()
    for i, (cam_name, img_size, extrinsic, intrinsic) in enumerate(zip(cam_names, img_sizes, extrinsics, intrinsics)):
        cam_dict = dict(
            name=cam_name,
            size=img_size.tolist(),
            matrix=intrinsic[0].tolist(),
            distortions=intrinsic[1].tolist(),
            rotation=extrinsic[:3].tolist(),
            translation=extrinsic[3:].tolist()
        )
        calibration_dict[f"cam_{i}"] = cam_dict
    calibration_dict["metadata"] = dict(adjusted=True, error=float(result.cost))

    with open(output_path, "w") as f:
        toml.dump(calibration_dict, f)


def read_calibration_toml(toml_path):
    """Read calibration data from TOML format"""
    with open(toml_path) as f:
        calibration_dict = toml.load(f)
    
    n_cams = len([k for k in calibration_dict.keys() if k.startswith("cam_")])
    cam_names = []
    img_sizes = []
    extrinsics = []
    intrinsics = []
    
    for i in range(n_cams):
        cam = calibration_dict[f"cam_{i}"]
        cam_names.append(cam["name"])
        img_sizes.append(np.array(cam["size"]))
        extrinsics.append(np.concatenate([cam["rotation"], cam["translation"]]))
        intrinsics.append((np.array(cam["matrix"]), np.array(cam["distortions"])))

    extrinsics = np.array(extrinsics)
        
    return cam_names, img_sizes, extrinsics, intrinsics


def load_calibration(calibration_dir: Path):
    calibration_paths = sorted(calibration_dir.glob("mc_calibration_output_*"))
    last_calibration_path = calibration_paths[-1]

    calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
    cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)
    print("Got calibration for the following cameras: ", cam_names)
    return cam_names, img_sizes, extrinsics, intrinsics, calib_toml_path


def get_view_from_filename_deeplabcut(filename: str) -> str:
    return filename.split("DLC")[0].split("_")[-1]


def get_view_from_filename_sleap(filename: str) -> str:
    raise NotImplementedError("SLEAP loading not supported yet")


def get_pose_files_dict(dir_path: str | Path, expected_views: tuple[str], software: str) -> dict[str, Path]:
    dir_path = Path(dir_path)
    suffix = None
    parsing_function = None
    if software == "DeepLabCut":
        suffix = "h5"
        parsing_function = get_view_from_filename_deeplabcut
    elif software == "SLEAP":
        suffix = "slp"
        parsing_function = get_view_from_filename_sleap
    else:
        raise ValueError(f"Non supported software: {software}")
    tracked_files = sorted(dir_path.glob(f"*mouse*{suffix}"))
    pprint(tracked_files)

    file_path_dict = {parsing_function(f.stem): f for f in tracked_files if "triangulated" not in f.name}
    file_path_dict = dict(sorted(file_path_dict.items()))
    pprint(file_path_dict)
    keys_tuple = tuple(file_path_dict.keys())
    pprint(keys_tuple)
    assert keys_tuple == expected_views, f"Expected views {expected_views}, got {keys_tuple}"
    return file_path_dict


def create_2d_ds(slp_files_dir: Path, expected_views: tuple[str], software: str, max_n_frames: int | None = None):
    # TODO read fps and skeleton
    file_path_dict = get_pose_files_dict(slp_files_dir, expected_views, software)

    # ds.attrs['fps'] = 'fps'
    

    # views_list = list(file_path_dict.keys())
    ds = from_multiview_files(file_path_dict, source_software=software)
    if max_n_frames is not None:
        ds = ds.isel(time=slice(0, max_n_frames))

    ds.attrs['skeleton'] = SKELETON

    return ds


def save_triangulated_ds(ds: xr.Dataset, valid_dir: Path):
    # Save the triangulated points using the directory name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = valid_dir / f"{valid_dir.name}_triangulated_points_{timestamp}.h5"

    ds.to_netcdf(save_path)

    return save_path


def load_triangulated_ds(save_path: Path):
    ds = xr.open_dataset(save_path)
    # TODO save in the file
    ds.attrs['skeleton'] = SKELETON
    return ds


if __name__ == "__main__":
    data_path = "/Users/vigji/Desktop/test_3d/M29/20250507/cricket/133050/multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021"
    from movement.filtering import filter_by_confidence
    expected_views = tuple(sorted(['mirror-bottom', 'mirror-left', 'mirror-top', 'central', 'mirror-right']))
    software = "DeepLabCut"
    
    ds = create_2d_ds(data_path, expected_views, software)

    ds = filter_by_confidence(ds.position, ds.confidence, threshold=0.5)
    # save_triangulated_ds(ds, Path(data_path))
    from matplotlib import pyplot as plt

    sel_central_back = ds.sel(view="central", keypoints="back_caudal")
    plt.figure()
    plt.plot(sel_central_back.sel(space="y"), lw=0.5, alpha=1, color="black")
    sel_central_back = ds.sel(view="central", keypoints="nose")
    plt.plot(sel_central_back.sel(space="y"), lw=0.5, alpha=1, color="red")
    plt.show()