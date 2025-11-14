import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import ssm
from tqdm import tqdm

FNAME_REGEX = re.compile(
    r"m(\d+)_s(\d+)_(cricket|object)\.xlsx$", re.IGNORECASE
)

def parse_filename(fname: str):
    m = FNAME_REGEX.match(fname)
    if not m:
        return None
    mouse_id = int(m.group(1))gg
    cond = m.group(3).lower()
    return mouse_id, cond

def session_index(name: str) -> int:
        """
        Extracts the integer after '_s' in filenames like 'm003_s007_cricket.xlsx'.
        Returns it as an int (e.g. 7).
        """
        m = re.search(r'_s(\d+)', name)
        if m is None:
            # fallback if pattern not found; put these at the end
            return 999999
        return int(m.group(1))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dir_features", required=True,)

    args = parser.parse_args()
    features_dir = Path(args.dir_features)
    feature_paths = sorted(features_dir.glob("*.xlsx"))
    feature_paths = [Path(path) for path in feature_paths]


    FEATURE_COLS_LINEAR = [
        "dist_head",
        "head_speed",
        "trunk_speed",
    ]

    # angular features (in degrees) -> we’ll convert to sin/cos
    ANGLE_COLS = [
        "facing_angle",
        "rel_angle_target",
    ]





