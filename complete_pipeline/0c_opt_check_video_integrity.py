#!/usr/bin/env python3
"""
Compare the frame-count of every cropped multicam video with its
parent (original) video.

Directory layout expected
-------------------------
<root>/
    multicam_video_<TIMESTAMP>.avi                ← original
    multicam_video_<TIMESTAMP>_cropped_<stamp>/   ← cropped folder
        multicam_video_<TIMESTAMP>_<view>.avi.mp4 ← cropped views

The script:
  * never modifies or creates files;
  * does **not** recurse – it searches only the first level of <root>;
  * prints one line per cropped file:
        OK          <orig#> <crop#>  <original>  <cropped>
        MISMATCH    …
        MISSING_ORIGINAL …
        ERROR       …

Usage
-----
python 0c_opt_check_video_integrity.py /path/to/folder [options]
"""

import argparse
from pathlib import Path
import re
import cv2
from pprint import pprint
import pandas as pd


def count_frames(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def check_once(root: Path, match_string: str, save_report: bool) -> None:
    print(root)
    all_cropped = sorted(list(root.glob(match_string)))

    parsing_results_dict_list = []
    all_ids = []
    for cropped_file in all_cropped:
        mouse = cropped_file.parts[-6]
        day = cropped_file.parts[-5]
        session = cropped_file.parts[-4]
        unique_id = mouse + "_" + day + "_" + session
        all_ids.append(unique_id)
        loadable = False
        original = None
        n_orig = None
        n_crop = None
        status = None
        
        try:
            try:
                original = next(cropped_file.parent.parent.glob(f"multicam_video_*.avi"))
            except StopIteration:
                print(f"MISSING_ORIGINAL\t{cropped_file}")
                status = "MISSING_ORIGINAL"
                continue
            n_orig = count_frames(original)
            n_crop = count_frames(cropped_file)
            status = "OK" if n_orig == n_crop else "MISMATCH"
            print(f"{unique_id}\t{status}\t{n_orig}\t{n_crop}\t{n_orig-n_crop}\t{original}\t{cropped_file}")
            loadable = True
        except Exception as e:
            print(f"ERROR\t{e}")
            status = "ERROR"

        result_dict = {
            "unique_id": unique_id,
            "loadable": loadable,
            "mouse": mouse,
            "day": day,
            "session": session,
            "cropped_file": cropped_file,
            "original": original,
            "n_orig": n_orig,
            "n_crop": n_crop,
            "status": status,
        }
        parsing_results_dict_list.append(result_dict)

    if save_report:
        df = pd.DataFrame(parsing_results_dict_list)
        df.to_csv("video_integrity_report.csv", index=False)
        print(f"Report saved to video_integrity_report.csv")

    
    all_ids = sorted(list(set(all_ids)))
    print(f"found {len(all_ids)} unique ids")
    pprint(all_ids)
    print(f"found {len(all_cropped)} cropped files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check video integrity by comparing frame counts between original and cropped videos.")
    
    parser.add_argument("folder", type=Path, help="Root folder containing video files")
    parser.add_argument("--config", type=Path, help="Optional config JSON file", default=None)
    parser.add_argument("--save-report", action="store_true", help="Save integrity report to CSV file")
    parser.add_argument("--match-string", type=str, 
                        default="M*/*/*/*/multicam_video_2025-*-*_cropped*/multicam_video_*_central.avi.mp4",
                        help="Glob pattern to match cropped video files")
    
    args = parser.parse_args()
    
    check_once(
        root=args.folder,
        match_string=args.match_string,
        save_report=args.save_report,
    )

