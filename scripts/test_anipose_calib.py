from aniposelib.boards import Checkerboard
import cv2


if __name__ == '__main__':
    from tqdm import tqdm
    board_shape = (5, 7)
    square_size = 12.5
    board = Checkerboard(board_shape[0], board_shape[1], square_size)

    # loop over individual frames in video and detect the board:
    video_path = "/Users/vigji/Desktop/test-anipose/cropped_calibration_vid/multicam_video_2024-08-03T16_36_58_central.avi.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    non_nan_frames = 0
    for _ in tqdm(range(total_frames // 10)):
        ret, frame = cap.read()
        if not ret:
            break
        corners, ids = board.detect_image(frame)
        if corners is not None:
            non_nan_frames += 1
            # board.manually_verify_board_detection(frame, corners)
        # print(corners, ids)
    print(f"Non-NaN frames: {non_nan_frames}")