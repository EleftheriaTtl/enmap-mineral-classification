#!/usr/bin/env python3
"""
02_enmap_process.py
===================

Step 2 of the EnMAP mineral-classification pipeline.

Runs `EnMAPProcessor` (from new_enmap.py) over every L2A archive that
step 01 dropped into `data/<mineral>/raw/`. NetCDFs land under
`data/<mineral>/netcdf/`.

Usage
-----
    python 02_enmap_process.py --mineral magnesite \
        --keep-only-swir --version 128 --apply-cloud-mask

You can still point at an arbitrary directory (legacy mode) instead of using
the per-mineral layout:

    python 02_enmap_process.py --base-dir "/Volumes/T7 Shield/sphalerite"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pipeline_utils as pu
from new_enmap import EnMAPProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EnMAP L2A processing on raw archives.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mineral", help="Mineral name; uses data/<mineral>/raw + data/<mineral>/netcdf.")
    g.add_argument("--base-dir", help="Legacy mode: explicit dir of EnMAP zips/dirs/tarballs.")
    ap.add_argument("--netcdf-dir", default=None, help="Override NetCDF output dir.")
    ap.add_argument("--keep-only-swir", action="store_true")
    ap.add_argument(
        "--version",
        default="v1",
        choices=["v1", "128", "58"],
        help="Band-dropping preset (matches EnMAPProcessor).",
    )
    ap.add_argument("--load-quality-layers", action="store_true")
    ap.add_argument("--apply-cloud-mask", action="store_true")
    args = ap.parse_args()

    if args.mineral:
        base_dir = pu.raw_dir(args.mineral)
        nc_dir = Path(args.netcdf_dir) if args.netcdf_dir else pu.netcdf_dir(args.mineral)
    else:
        base_dir = Path(args.base_dir)
        nc_dir = Path(args.netcdf_dir) if args.netcdf_dir else (base_dir / "netcdf_outputs")

    if not base_dir.exists():
        print(f"[process] base dir not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    p = EnMAPProcessor(
        base_dir=str(base_dir),
        save_netcdf=True,
        netcdf_dir=str(nc_dir),
        load_quality_layers=args.load_quality_layers,
        apply_cloud_mask=args.apply_cloud_mask,
        keep_only_swir=bool(args.keep_only_swir),
        version=args.version,
    )
    out = p.process_all_available_data()
    summary = {k: len(v) for k, v in out.items()}
    print(f"[process] DONE -> {nc_dir}")
    print(summary)


if __name__ == "__main__":
    main()
