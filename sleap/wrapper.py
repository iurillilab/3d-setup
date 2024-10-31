import subprocess
import sys

log_filename = "inference_run.log"

with open(log_filename, "w") as log_file:
    process = subprocess.Popen(
        [sys.executable, "model_inference.py"],
        stdout=log_file,
        stderr=log_file,
        universal_newlines=True,
    )
    process.communicate()
