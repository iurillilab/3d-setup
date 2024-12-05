# ================================
# Checkerboard detection
# ================================
# from aniposelib.boards import Checkerboard
# import cv2

# if __name__ == '__main__':
#     from tqdm import tqdm
#     board_shape = (5, 7)
#     square_size = 12.5
#     board = Checkerboard(board_shape[0], board_shape[1], square_size)

#     # loop over individual frames in video and detect the board:
#     video_path = "/Users/vigji/Desktop/test-anipose/cropped_calibration_vid/multicam_video_2024-08-03T16_36_58_central.avi.mp4"
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     non_nan_frames = 0
#     for _ in tqdm(range(total_frames // 10)):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         corners, ids = board.detect_image(frame)
#         if corners is not None:
#             non_nan_frames += 1
#             # board.manually_verify_board_detection(frame, corners)
#         # print(corners, ids)
#     print(f"Non-NaN frames: {non_nan_frames}")

# %%
# ================================
# Calibration
# ================================
# from threed_utils.anipose.calibrate import process_session_core
# from threed_utils.anipose.common import get_cam_name
# import re

# from pathlib import Path
# calib_config = dict(board_type="checkerboard",
#                 board_size=(5, 7),
#                 board_square_side_length=12.5,
#                 animal_calibration=False,
#                 calibration_init=None,
#                 fisheye=False)

# triang_config = {
#     "cam_regex": r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([\w-]+)\.\w+(?:\.\w+)?$"
# }
# manual_verification_config = dict(manually_verify=False)

# config = dict(calibration=calib_config, 
#               triangulation=triang_config,
#               manual_verification=manual_verification_config,
#               )
# data_dir = Path("/Users/vigji/Desktop/test-anipose/cropped_calibration_vid")
# videos = [f for f in data_dir.glob("*.mp4")]
# print(videos)
# process_session_core(config, videos, str(data_dir / "anipose_calib"))


# %%
# ================================
# Triangulation
# ================================

# from threed_utils.anipose.common import get_cam_name
from threed_utils.anipose.triangulate import triangulate_core, CameraGroup
from movement.io.load_poses import from_file, from_multiview_files
import pickle
import re
import numpy as np
import xarray as xr
import pandas as pd


from pathlib import Path
calib_config = dict(board_type="checkerboard",
                board_size=(5, 7),
                board_square_side_length=12.5,
                animal_calibration=False,
                calibration_init=None,
                fisheye=False)

triang_config = {
    "cam_regex": r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([\w-]+)\.\w+(?:\.\w+)+$",
    "ransac": False,
    "optim": False,
    "optim_chunking": True,
    "optim_chunking_size": 100,
    "score_threshold": 0.0,
}
manual_verification_config = dict(manually_verify=False)

config = dict(calibration=calib_config, 
              triangulation=triang_config,
              manual_verification=manual_verification_config,
              )
data_dir = Path("/Users/vigji/Desktop/test-anipose")
calibration_dir = data_dir / "cropped_calibration_vid" / "anipose_calib"
slp_files_dir = data_dir / "test_slp_files"
slp_files = list(slp_files_dir.glob("*.slp"))
file_path_dict = {re.search(triang_config['cam_regex'], str(f.name)).groups()[0]: f for f in slp_files}
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

time_slice = slice(0, 1000)
ds = xr.concat(dataset_list, dim=new_coord_views).sel(time=time_slice, individuals="individual_0", drop=True)

bodyparts = list(ds.coords["keypoints"].values)

print(ds.position.shape, ds.confidence.shape, bodyparts)

calib_fname = calibration_dir / 'calibration.toml'
cgroup = CameraGroup.load(calib_fname)

output_fname = data_dir / "test_triangulation.csv"
triangulate_core(config, ds.position.values, ds.confidence.values, bodyparts, cgroup, output_fname)

triang_ds = pd.read_csv(output_fname)
print(triang_ds.head())


from matplotlib import pyplot as plt

plt.plot(triang_ds['blimbmid_x'], triang_ds['blimbmid_y'])
plt.show()

# from_file(filename, software="SLEAP")

# pickled_detections = data_dir / "anipose_calib" / "detections.pickle"
# with open(pickled_detections, "rb") as f:
#     detections = pickle.load(f)
# print(len(detections), len(detections[0]), detections[0][0].keys(), detections[0][0]['ids'].shape, detections[0][0]['corners'].shape)
# for i in range(len(detections[0][:10])):
#     print(detections[0][i]['ids'])
# # videos = [f for f in data_dir.glob("*.mp4")]

# all_points_raw is array of shape (n_cams, n_frames, n_joints, 2)
# all_scores is array of shape (n_cams, n_frames, n_joints), I assum


# or video in videos:
    # print("Looking at video: ", video.name)
    # matching multicam_video_2024-08-03T16_36_58_mirror-right.avi.mp4
    # regex = "multicam_video_*"
    # match = re.search(triang_config['cam_regex'], str(video.name))
    # print(match)
    # print(match.groups())
            

    # print(get_cam_name(config, str(video)))

# %%
# old triangulation:

# import pickle
# import flammkuchen as fl
# from tqdm import tqdm

# data_dir = Path("/Users/vigji/Desktop/test-anipose/cropped_calibration_vid")
# old_pickle_fname = data_dir / "anipose_calib" / "detections_orig.pickle"
# with open(old_pickle_fname, "rb") as f:
#     pickle_sample = pickle.load(f)

# all_flammkuchen_files = list(data_dir.glob("*.detections.h5"))

# full_pickle_imitation = []  
# for f in tqdm(all_flammkuchen_files):
#     fl_sample = fl.load(f)
#     imitating_list = []
#     for i in range(len(fl_sample["frame_ixs"])):
#         frame_dict = {}
#         frame_dict["framenum"] = (0, fl_sample["frame_ixs"][i])
#         frame_dict["corners"] = fl_sample["uvs"][i][:, np.newaxis, :]
#         frame_dict["ids"] = np.arange(len(fl_sample["uvs"][i]))
#         frame_dict["filled"] = fl_sample["uvs"][i][:, np.newaxis, :]
#         imitating_list.append(frame_dict)
#     full_pickle_imitation.append(imitating_list)

# with open(old_pickle_fname.parent / "detections.pickle", "wb") as f:
#     pickle.dump(full_pickle_imitation, f)
# # %%


#     # for (im_key, im_val), (pickle_key, pickle_val) in zip(imitating_list[0].items(), pickle_sample[0][0].items()):
#     #     print(im_key, pickle_key)
#     #     assert im_key == pickle_key
#     #     try:
#     #         print(im_val.shape, pickle_val.shape)
#     #         assert im_val.shape == pickle_val.shape
#     #     except AttributeError:
#     #         print(im_val, pickle_val)

    
#     # assert np.allclose(im_val, pickle_val)
# # %%
# fl_sample["frame_ixs"]
# # %%
# fl_sample["img_size"]
# # %%
# fl_sample["qc_data"].shape
# # %%
# fl_sample["uvs"].shape
# # %%
# pickle_sample[0][0].keys()
# # %%
# len(pickle_sample)
# # %%
# pickle_sample[0][2]["framenum"]
# # %%
# pickle_sample[0][0]["corners"].shape
# # %%
# pickle_sample[0][0]["corners"].ids
# # %%
# pickle_sample[0][0]["filled"].shape

# # %%
# plt.figure()
# plt.scatter(pickle_sample[0][0]["corners"][:, 0, 0], pickle_sample[0][0]["corners"][:, 0, 1])
# # plt.scatter(pickle_sample[0][0]["filled"][:, 0, 0], pickle_sample[0][0]["filled"][:, 0, 1])

# plt.show()
# # %%
