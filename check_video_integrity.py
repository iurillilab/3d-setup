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
python check_frames_norecurse.py /path/to/cricket/133050
"""

from pathlib import Path
import re
import cv2
from pprint import pprint


CROPPED_FILE_RE = re.compile(
    r"multicam_video_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_[A-Za-z0-9]+\.avi\.mp4$"
)


def count_frames(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def check_once() -> None:
    # root = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_video_recordings")
    root = Path("/Volumes/COMPATIBLE/nas_mirror")
    print(root)
    all_cropped = sorted(list(root.glob("M*/*/*/*/multicam_video_2025-*-*_cropped_*/multicam_video_*_central.avi.mp4")))

    all_ids = []
    for cropped_file in all_cropped:
        mouse = cropped_file.parts[-6]
        day = cropped_file.parts[-5]
        session = cropped_file.parts[-4]
        unique_id = mouse + "_" + day + "_" + session
        all_ids.append(unique_id)
        try:
            try:
                original = next(cropped_file.parent.parent.glob(f"multicam_video_*.avi"))
            except StopIteration:
                print(f"MISSING_ORIGINAL\t{cropped_file}")
                continue
            n_orig = count_frames(original)
            n_crop = count_frames(cropped_file)
            status = "OK" if n_orig == n_crop else "MISMATCH"
            print(f"{unique_id}\t{status}\t{n_orig}\t{n_crop}\t{n_orig-n_crop}\t{original}\t{cropped_file}")
        except Exception as e:
            print(f"ERROR\t{e}")
    all_ids = sorted(list(set(all_ids)))
    print(f"found {len(all_ids)} unique ids")
    pprint(all_ids)
    print(f"found {len(all_cropped)} cropped files")


if __name__ == "__main__":
    check_once()
