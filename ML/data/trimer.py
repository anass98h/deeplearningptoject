#!/usr/bin/env python3
"""
sync_frames.py
--------------

Delete rows from each <example>.csv (in output_poses/) whose FrameNo is **not**
present in the corresponding <example>_kinect.csv (in kinect_good_preprocessed/).

• A backup of the original <example>.csv is saved as <example>.csv.bak
  unless you pass --no-backup.

Usage
-----
    python sync_frames.py                      # uses default folder names
    python sync_frames.py --kinect_dir my_kinect \
                          --output_dir my_poses
    python sync_frames.py --no-backup          # overwrite without .bak files
"""
from pathlib import Path
import argparse
import sys

import pandas as pd


def sync_files(kinect_dir: Path,
               output_dir: Path,
               backup: bool = True,
               kinect_suffix: str = "_kinect.csv") -> None:
    """
    Iterate over every CSV in `output_dir` and trim it so that its FrameNo set
    equals the one found in the matching _kinect.csv file.
    """
    if not kinect_dir.is_dir() or not output_dir.is_dir():
        sys.exit("❌  One or both folder paths do not exist.")

    for out_file in output_dir.glob("*.csv"):
        stem = out_file.stem            # "example"
        kin_file = kinect_dir / f"{stem}{kinect_suffix}"

        if not kin_file.exists():
            print(f"⚠️  No Kinect file for {out_file.name}; skipping.")
            continue

        # Read only the FrameNo column from the Kinect file for speed.
        k_frames = (
            pd.read_csv(kin_file, usecols=["FrameNo"])
            .squeeze("columns")
            .astype(int)
            .unique()
        )

        # Load the full output CSV, filter rows, and save.
        out_df = pd.read_csv(out_file)
        before = len(out_df)

        out_df = out_df[out_df["FrameNo"].isin(k_frames)].copy()
        out_df.sort_values("FrameNo", inplace=True)
        out_df.reset_index(drop=True, inplace=True)
        after = len(out_df)

        if backup:
            out_file.rename(out_file.with_suffix(".csv.bak"))

        out_df.to_csv(out_file, index=False)
        print(f"✅  {out_file.name}: removed {before - after} rows (kept {after}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim output_poses CSV files to match FrameNo values in the "
                    "_kinect.csv counterparts."
    )
    parser.add_argument("--kinect_dir", type=Path,
                        default="kinect_good_preprocessed",
                        help="Folder containing <example>_kinect.csv files.")
    parser.add_argument("--output_dir", type=Path,
                        default="output_poses",
                        help="Folder containing <example>.csv files to be trimmed.")
    parser.add_argument("--no-backup", dest="backup", action="store_false",
                        help="Disable creation of a .bak backup.")
    args = parser.parse_args()

    sync_files(args.kinect_dir, args.output_dir, args.backup)


if __name__ == "__main__":
    main()
