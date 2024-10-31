import json
import logging
import os
import subprocess
import sys
import threading

# find all configs
backbone_folder_path_list = [
    r"C:\Users\SNeurobiology\code\3d-setup\sleap\models\241007_120850.single_instance.n=500\initial_config.json",
    r"C:\Users\SNeurobiology\code\3d-setup\sleap\models\241007_153439.single_instance.n=501\initial_config501.json",
]
models = ["base", "high_res"]
list_files = []
labels_path = r"D:\SLEAP_models\test\labels.v001.slp"
out_results_path = r"D:\SLEAP_models\results_multi_training"

# loop through the configs and change the batch size and plateau values

params = []

batch_size = [4, 8, 16, 32]
stride = [8, 16, 32]

for i in batch_size:
    for j in stride:
        params.append((i, j))


print("Avialable params:", params)


for k in range(len(params)):
    print(params[k])
    batch_size = params[k][0]
    stride = params[k][1]
    for backbone_folder_path, model in zip(backbone_folder_path_list, models):
        with open(backbone_folder_path, "r") as backbone_file:
            data = json.load(backbone_file)

        original_json = data
        original_json["optimization"]["batch_size"] = batch_size
        original_json["model"]["backbone"]["max_stride"] = stride
        original_json["outputs"][
            "run_name_prefix"
        ] = f"{model}_batch{batch_size}_stride{stride}"
        original_json["outputs"]["runs_folder"] = out_results_path
        new_file_name = rf"C:\Users\SNeurobiology\code\3d-setup\sleap\models\training{model}{batch_size}_stride{stride}.json"
        # save the json to a new json file
        with open(new_file_name, "w") as f:
            json.dump(original_json, f)
            list_files.append(f.name)
            print(list_files)
        print(f"successfully created {new_file_name}")

# train the models
for model in list_files:
    call = f"sleap-train {model} {labels_path}"
    try:
        # os.system(call)
        subprocess.run(call, shell=True, check=True)
        print(f"successfully trained {model}")
    except subprocess.CalledProcessError as e:
        print(f"Error running comman {call}, error: {e}")
