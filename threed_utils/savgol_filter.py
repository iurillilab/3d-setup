"""Script to apply Savitzky-Golay filter to a triangulations."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from scipy.signal import savgol_filter
from tqdm import tqdm

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input  file containing the triangulation data.",
    )
    PARSER.add_argument("--keypoint_cols", type=list | str, default="All")
    args = PARSER.parse_args()
    INPUT_PATH = Path(args.input_path)
