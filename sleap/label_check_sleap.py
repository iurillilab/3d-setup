import json
import os
import subprocess
import logging
import sys
import threading
from pathlib import Path
list_files = []
config_path = r"D:\SLEAP_models\results_multi_training\base_batch32_stride16241007_120850.single_instance.n=500\initial_config.json"
output_results = r"D:\SLEAP_models\results_multi_training\check_labels_influence"
labels_path = r'D:\SLEAP_models\test\labels.v001.slp'

p = Path(r'D:\SLEAP_models\results_multi_training')
model_name = 'base_batch32_stride16241007_120850.single_instance.n=500'


with open(config_path, "r") as config:
    data = json.load(config)

original_json = data
total_train_labels = len(original_json['data']['labels']['training_inds'])

step = 50

for i in range(0, total_train_labels, step):
    with open(config_path, "r") as config:
        data = json.load(config)

    original_json = data
    train_labels = original_json['data']['labels']['training_inds'][0:i +step]
    original_json['data']['labels']['training_inds'] = train_labels
    original_json["outputs"]["run_name_prefix"] = f"best_model_n{i}"
    original_json["outputs"]["runs_folder"] = output_results
    new_file_name = rf"C:\Users\SNeurobiology\code\3d-setup\sleap\models\trainingbest_model_n{i}.json"
    print(f'len of new training index : {len(train_labels)}, for model: {i}')
    with open(new_file_name, "w") as f:
        json.dump(original_json, f)
        list_files.append(f.name)
        print(f"successfully created {new_file_name}")



for model in list_files:
     call = f"sleap-train {model} {labels_path}"
     try:
          #os.system(call)
          subprocess.run(call, shell=True, check=True)
          print(f"successfully trained {model}")
     except subprocess.CalledProcessError as e:
          print(f"Error running comman {call}, error: {e}")

