import datetime
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Set

import numpy as np
from tqdm import tqdm

from utils import crop_all_views

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

MODELS_LOCATIONS = {
    "side":   r"D:\SLEAP_models\SLEAP_side_models\models\250314_091459.single_instance.n=659",
    "bottom": r"D:\SLEAP_models\SLEAP_bottom_model\models\250116_131653.single_instance.n=416",
}

MODELS_MAP_TO_VIEW = {
    "side":   ["mirror-top", "mirror-bottom", "mirror-left", "mirror-right"],
    "bottom": ["central"],
}

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------


def run_inference(video: Path, model: Path) -> None:
    """
    Run SLEAP inference on *video* with *model* and store results next to the video.
    """
    out_path = video.with_suffix("").with_suffix(".predictions.slp")

    # NOTE: avoid spawning a full shell + `conda activate` for every video.
    # Call the sleap executable directly; ensure the script is launched from
    # the correct env or set SLEAP_EXE in the OS environment.
    sleap_exe = os.environ.get("SLEAP_EXE", "sleap-track")

    cmd = [
        sleap_exe,
        "-m", str(model),
        "-o", str(out_path),
        str(video)
    ]
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SLEAP failed on {video} → {e}")


def _cached_cropped_stems(folder: Path) -> Set[str]:
    """
    Return the *stems* of already-cropped videos once, so we don't re-walk SMB dirs.
    """
    pattern = re.compile(r"^(?P<stem>.+)_cropped_")
    stems = set()
    for p in folder.rglob("*_cropped_*"):
        m = pattern.match(p.name)
        if m:
            stems.add(m.group("stem"))
    return stems


def process_videos_in_folder(folder: Path,
                             json_file: Path,
                             timestamp: str,
                             skip_existing: bool = True) -> None:
    """
    Find all AVI videos in *folder*, crop each view, optionally run inference.
    Skips files that already have a matching *_cropped_* sibling when
    *skip_existing* is True.
    """
    # deterministic order → easier to compare timings
    avi_files = sorted(folder.rglob("*.avi"))

    # ignore already-split views (mirror-*, central, …)
    EXCLUDE = {"central", "mirror-top", "mirror-bottom", "mirror-left", "mirror-right"}
    avi_files = [f for f in avi_files if not any(tag in f.stem for tag in EXCLUDE)]

    # prime the cache once – avoids O(N²) directory walks over SMB
    done_stems = _cached_cropped_stems(folder) if skip_existing else set()

    for avi_file in tqdm(avi_files, desc="cropping"):
        if skip_existing and avi_file.stem in done_stems:
            continue

        out_dir = avi_file.parent / f"{avi_file.stem}_cropped_{timestamp}"
        cropped_files = crop_all_views(avi_file, out_dir, json_file, verbose=False)
        cropped_files = [f.result() for f in cropped_files]

        done_stems.add(avi_file.stem)          # keep the cache up-to-date

        # ------------------------------------------------------------------
        # OPTIONAL: run SLEAP inference on freshly cropped clips
        # ------------------------------------------------------------------
        # for model_name, views in MODELS_MAP_TO_VIEW.items():
        #     videos = [vf for vf in cropped_files
        #               if any(view in vf.name for view in views)]
        #     for vf in videos:
        #         run_inference(vf, MODELS_LOCATIONS[model_name])


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop all views in AVI files.")
    parser.add_argument("folder",      type=Path, help="Folder containing AVI files.")
    parser.add_argument("json_file",   type=Path, help="JSON with crop params.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip processing if *_cropped_* exists.")

    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    process_videos_in_folder(
        folder=args.folder,
        json_file=args.json_file,
        timestamp=timestamp,
        skip_existing=not args.skip_existing,
    )
