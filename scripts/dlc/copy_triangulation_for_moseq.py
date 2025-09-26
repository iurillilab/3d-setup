#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import re
import sys
import shutil
import argparse
import hashlib
from tqdm import tqdm
STAMP_RE = re.compile(r"_triangulated_points_(\d{8}-\d{6})\.h5$")

def parse_stamp(p: Path):
    m = STAMP_RE.search(p.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    except ValueError:
        return None

def unique_dest(target: Path, src: Path, rel_hint: Path) -> Path:
    """
    Return a destination path in target (flat) that avoids collisions.
    If target/filename exists, append _{hash8} before the .h5 extension,
    using a stable hash of the source file's relative path.
    """
    base = src.name
    dest = target / base
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix  # ".h5"
    h = hashlib.md5(str(rel_hint).encode("utf-8")).hexdigest()[:8]
    return target / f"{stem}_{h}{suffix}"

def safe_copyfile(src: Path, dst: Path):
    # Ensure the target dir exists (but we don't create subdirs beyond target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # data only; avoids metadata/permission issues on WSL

def main():
    ap = argparse.ArgumentParser(description="Copy latest triangulated .h5 per directory (flat copies).")
    ap.add_argument("source", type=Path, help="Base source dir (e.g., /mnt/y/nas_mirror/M30)")
    ap.add_argument("target", type=Path, help="Destination dir (flat; only files placed here)")
    ap.add_argument("--allow", nargs="+", default=["cricket", "roach"],
                    help="Session keywords to include (default: cricket roach)")
    ap.add_argument("--preserve-structure", action="store_true",
                    help="OPTIONAL: mirror source subdirs (not default).")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    args.target.mkdir(parents=True, exist_ok=True)

    pattern = "*_triangulated_points_*.h5"
    latest_by_dir = {}
    for f in args.source.rglob(pattern):
        if not any(k in f.parts for k in args.allow):
            continue
        stamp = parse_stamp(f)
        sort_key = stamp or datetime.fromtimestamp(f.stat().st_mtime)
        parent = f.parent
        prev = latest_by_dir.get(parent)
        if (prev is None) or (sort_key > prev[0]):
            latest_by_dir[parent] = (sort_key, f)

    selected = [v[1] for _, v in sorted(latest_by_dir.items(), key=lambda kv: kv[0].as_posix())]

    copied = 0
    for src_file in tqdm(selected):
        rel_hint = src_file.relative_to(args.source)
        if args.preserve_structure:
            # Mirror structure: create subdirs under target
            dest_file = args.target / rel_hint
            dest_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Flat: put just the .h5 in target, no subdirs; dedupe if needed
            dest_file = unique_dest(args.target, src_file, rel_hint)

        try:
            safe_copyfile(src_file, dest_file)
            copied += 1
        except Exception as e:
            print(f"Failed to copy {src_file} -> {dest_file}: {e}", file=sys.stderr)

    print(f"\nTotal copied: {copied} (out of {len(selected)})")

if __name__ == "__main__":
    main()

