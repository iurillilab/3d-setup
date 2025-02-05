import numpy as np
from movement.io.load_poses import from_numpy
import toml


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
    confidence_array = np.zeros((n_frames, 1, n_keypoints))
    for i, kp in enumerate(keypoint_names):
        for j, coord in enumerate(['x', 'y', 'z']):
            position_array[:, 0, i, j] = triang_df[f'{kp}_{coord}']
        confidence_array[:, 0, i] = triang_df[f'{kp}_score']

    individual_names = [individual_name]
    position_array = position_array.transpose(0, 3, 2, 1)
    confidence_array = confidence_array.transpose(0, 2, 1)

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