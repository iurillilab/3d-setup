from pathlib import Path
from tqdm import tqdm
from pprint import pprint
path = Path("/mnt/d/nas_mirror")
models_path = Path('/content/drive/MyDrive/dlc3_models')

DRYRUN = False

all_to_process = list(path.glob("M*/*/*/*/*/*central*.mp4"))

sessions = ["cricket", "object"] # ["object", "cricket", "roach"]
retrace = [] #["cricket",]

shuffle_dict = {"object":3,
                "mouse":2,
                "cricket":8}

config_files_dict = {"object": "/mnt/d/DLC_models/dlc3_object-YaduLuigi-2025-06-10/config.yaml",
                     "cricket": "/mnt/d/DLC_models/dlc3_cricket-YaduLuigi-2025-06-10/config.yaml",
                     }

for config_file in config_files_dict.values():
    assert Path(config_file).exists()

# for session in sessions:
  #print(session, "folder")
  #videos_to_analyze = list((videos_path / session ).glob("*.mp4"))
  # pprint(videos_to_analyze)
  #session_config_path = next(models_path.glob(f'*{session}*/config.yaml'))

  # for modelname, model_config in zip([session], [session_config_path]):  # "mouse", mouse_model_config, 
    # print("===", (model_config), "===")
actually_to_process = []
all_videos = sorted(list(path.glob("M*/*/*/*/*/*central*.mp4")))
for video_path in all_videos:
    session = video_path.parts[-4]
    for modelname in [session,]:  # add mouse here
      if len(list(video_path.parent.glob(f"{video_path.stem}*{modelname}*.h5"))) > 0 and not modelname in retrace:
        print("already inferred", video_path)
        pass
      else:
        actually_to_process.append((str(video_path), modelname))

pprint(actually_to_process)
    # Analyze videos using the trained model from shuffle 7

bar = tqdm if not DRYRUN else lambda x: x

for filename, modelname in bar(actually_to_process):
    if DRYRUN:
        print(modelname, filename)
    else:
        import deeplabcut
        deeplabcut.analyze_videos(config_files_dict[modelname], [str(filename)], batchsize=1, shuffle=shuffle_dict[modelname])