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
    Skips files that already have a matching *_cropped-vX_* sibling when
    *skip_existing* is True.
    """
    crop_header_string = "cropped-v2"
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

        out_dir = avi_file.parent / f"{avi_file.stem}_{crop_header_string}_{timestamp}"
        cropped_files = crop_all_views(avi_file, out_dir, json_file, verbose=False)
        cropped_files = [f.result() for f in cropped_files]

        done_stems.add(avi_file.stem)          # keep the cache up-to-date


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop all views in AVI files.")
    parser.add_argument("folder",      type=Path, help="Folder containing AVI files.")
    parser.add_argument("--json_file", type=Path, help="JSON with crop params. Defaults to crop_params.json in folder", default=None)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip processing if *_cropped_* exists.")

    args = parser.parse_args()

    if not args.json_file:
        args.json_file = Path(args.folder) / "cropping_params.json" # TODO: make this a default

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    process_videos_in_folder(
        folder=args.folder,
        json_file=args.json_file,
        timestamp=timestamp,
        skip_existing=args.skip_existing,
    )
