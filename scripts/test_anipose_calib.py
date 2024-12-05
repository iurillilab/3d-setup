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

# process_session_core(config, videos, str(data_dir / "anipose_calib"))


# ================================
# Triangulation
# ================================

from threed_utils.anipose.common import get_cam_name
from threed_utils.anipose.triangulate import triangulate_core, CameraGroup
import pickle


from pathlib import Path
calib_config = dict(board_type="checkerboard",
                board_size=(5, 7),
                board_square_side_length=12.5,
                animal_calibration=False,
                calibration_init=None,
                fisheye=False)

triang_config = {
    "cam_regex": r"multicam_video_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_([\w-]+)\.\w+(?:\.\w+)?$"
}
manual_verification_config = dict(manually_verify=False)

config = dict(calibration=calib_config, 
              triangulation=triang_config,
              manual_verification=manual_verification_config,
              )
data_dir = Path("/Users/vigji/Desktop/test-anipose/cropped_calibration_vid")
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
