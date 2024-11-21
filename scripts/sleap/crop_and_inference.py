from run_inference import run_inference, get_video_paths
import os
import pathlib
import subprocess

main_path = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting"
crop_json_file = None


#crop all the video
subprocess.run(['python', '0b_process_videos.py', main_path, crop_json_file])



# get the path diveded into side and bottom for all videos

main_path = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting"
side_paths, bottom_paths = get_video_paths(main_path)

# run side model on all the side videos
side_model = r"D:\SLEAP_models\SLEAP_side_models\models\241007_120850.single_instance.n=500"

run_inference(side_paths, side_model)

# run bottom model on all the bottom videos
bottom_model = r"D:\SLEAP_models\SLEAP_bottom_model\models\241106_104724.single_instance.n=161"
run_inference(bottom_paths, bottom_model)

