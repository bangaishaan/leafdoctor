"""Zip data/processed/ into leafdoctor_data.zip for upload as a private Kaggle dataset.

The zip holds the processed image folders at its root, plus class_map.csv and
split_manifest.csv, so the extracted Kaggle dataset is a complete --data-root
for src/train.py:

  <data-root>/plantvillage/<config>/<split>/<class_id>/*.jpg
  <data-root>/plantdoc/<field_split>/<class_id>/*
  <data-root>/class_map.csv
  <data-root>/split_manifest.csv

Every file listed in split_manifest.csv is checked before packing. Images are
stored without recompression (JPEGs do not shrink), which keeps packing fast.

Upload: kaggle.com -> Datasets -> New Dataset -> upload the zip -> keep it Private.

Example:
  python scripts/pack_for_kaggle.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--class-map", type=Path, default=REPO_ROOT / "data" / "class_map.csv")
    p.add_argument("--split-manifest", type=Path, default=REPO_ROOT / "data" / "split_manifest.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "leafdoctor_data.zip")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-entry-len", type=int, default=100,
                   help="fail if any zip entry path is longer than this (Kaggle rejects entries over 248 bytes)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for f in (args.processed_dir, args.class_map, args.split_manifest):
        if not f.exists():
            sys.exit(f"{f} not found; run src/prepare_data.py first")
    if args.out.exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite to replace it")

    with open(args.split_manifest, newline="", encoding="utf-8") as f:
        paths = [r["path"] for r in csv.DictReader(f)]
    missing = [p for p in paths if not (args.processed_dir / p).is_file()]
    if missing:
        sys.exit(f"{len(missing)} files in {args.split_manifest.name} are missing from {args.processed_dir}, "
                 f"e.g. {missing[:3]}; rerun src/prepare_data.py")
    longest = max(paths, key=lambda p: len(p.encode("utf-8")))
    if len(longest.encode("utf-8")) > args.max_entry_len:
        sys.exit(f"zip entry too long ({len(longest.encode('utf-8'))} > {args.max_entry_len} bytes): {longest}; "
                 "rerun src/prepare_data.py, which writes short hashed file names")
    print(f"longest entry: {len(longest.encode('utf-8'))} bytes ({longest})")

    tmp = args.out.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.write(args.class_map, "class_map.csv", compress_type=zipfile.ZIP_DEFLATED)
        zf.write(args.split_manifest, "split_manifest.csv", compress_type=zipfile.ZIP_DEFLATED)
        for i, rel in enumerate(paths, 1):
            zf.write(args.processed_dir / rel, rel)
            if i % 10000 == 0:
                print(f"packed {i}/{len(paths)}", flush=True)
    tmp.replace(args.out)

    with zipfile.ZipFile(args.out) as zf:
        names = set(zf.namelist())
    lost = [p for p in paths if p not in names]
    if lost:
        sys.exit(f"zip check failed: {len(lost)} images missing, e.g. {lost[:3]}")
    size_gb = args.out.stat().st_size / 1e9
    print(f"wrote {args.out} ({len(paths)} images + class_map.csv + split_manifest.csv, {size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
