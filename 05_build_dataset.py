#!/usr/bin/env python3
"""
05_build_dataset.py
===================

Step 5 of the EnMAP mineral-classification pipeline.

Pairs every clipped chip from `data/<mineral>/chips/<mine>/*.nc` with its
OSM mask from `data/<mineral>/masks/<mine>/<chip_stem>_osm_mask.npy` and
writes an ML-ready dataset under `data/<mineral>/dataset/`:

    data/<mineral>/dataset/
        manifest.csv                       # one row per sample, split=unassigned
        samples/<chip_id>.npz              # X, y, wavelength, lat, lon, mine, ...

Each .npz contains:

    X            float32   (B, H, W)       reflectance (B = #bands kept)
    y            uint8     (H, W)          0=background, 1=mineral-area (OSM)
    wavelength   float32   (B,)            band centers in nm

And these scalar attrs (saved as 0-d arrays so np.load can read them):

    chip_id, mineral, mine_name, scene_id, lat, lon, height, width

Usage
-----
    python 05_build_dataset.py --mineral magnesite               # 128x128 default
    python 05_build_dataset.py --mineral magnesite --chip-size 64
    python 05_build_dataset.py --mineral magnesite --any-size    # don't filter

Each sample is independent; you can stream them via a torch Dataset that
just iterates `manifest.csv` and `np.load(npz_path)`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import pipeline_utils as pu


def _load_chip(nc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, wavelength) where X is (B,H,W) float32, wavelength is (B,) float32."""
    ds = xr.open_dataset(nc_path, engine="netcdf4", mask_and_scale=False)
    X = ds["reflectance"].values.astype(np.float32)  # (B,H,W) — built that way upstream
    wl = ds["wavelength"].values.astype(np.float32)
    return X, wl


def _load_mask(npy_path: Path, *, h: int, w: int) -> Optional[np.ndarray]:
    if not npy_path.exists():
        return None
    arr = np.load(npy_path)
    if arr.shape != (h, w):
        return None
    return arr.astype(np.uint8)


def _scene_id(chip_path: Path, mine_safe: str) -> str:
    # chip stem: <mine_safe>_<scene_stem>.nc
    stem = chip_path.stem
    if stem.startswith(mine_safe + "_"):
        return stem[len(mine_safe) + 1 :]
    return stem


def build(
    *,
    mineral: str,
    chip_size: Optional[int],
    out_dir: Path,
    overwrite: bool,
) -> int:
    chips_root = pu.chips_dir(mineral, create=False)
    masks_root = pu.masks_dir(mineral, create=False)

    if not chips_root.exists():
        raise FileNotFoundError(f"No chips dir: {chips_root}")
    if not masks_root.exists():
        raise FileNotFoundError(
            f"No masks dir: {masks_root} -- run 04_osm_mask_batch.py first."
        )

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    chip_files = sorted(chips_root.glob("*/*.nc"))
    if not chip_files:
        raise FileNotFoundError(f"No chip NetCDFs under {chips_root}")

    rows: list[dict] = []
    n_ok = 0
    n_skip_size = 0
    n_skip_mask = 0

    for nc_path in chip_files:
        mine_safe = nc_path.parent.name
        scene_id = _scene_id(nc_path, mine_safe)

        try:
            X, wl = _load_chip(nc_path)
        except Exception as e:
            print(f"[dataset] FAIL load {nc_path}: {e}", file=sys.stderr)
            continue

        B, H, W = X.shape
        if chip_size is not None and (H != chip_size or W != chip_size):
            n_skip_size += 1
            print(
                f"[dataset] SKIP wrong-size {mine_safe}/{nc_path.name} ({H}x{W} != {chip_size})",
                file=sys.stderr,
            )
            continue

        mask_path = masks_root / mine_safe / f"{nc_path.stem}_osm_mask.npy"
        y = _load_mask(mask_path, h=H, w=W)
        if y is None:
            # try legacy side-by-side mask location too
            legacy_mask = nc_path.with_suffix("").as_posix() + "_osm_mask.npy"
            y = _load_mask(Path(legacy_mask), h=H, w=W)
        if y is None:
            n_skip_mask += 1
            print(f"[dataset] SKIP no/mismatched mask {mask_path}", file=sys.stderr)
            continue

        chip_id = f"{mine_safe}__{scene_id}"
        out_npz = samples_dir / f"{chip_id}.npz"
        if out_npz.exists() and not overwrite:
            pass
        else:
            np.savez_compressed(
                out_npz,
                X=X,
                y=y,
                wavelength=wl,
                chip_id=np.asarray(chip_id),
                mineral=np.asarray(mineral),
                mine_name=np.asarray(mine_safe),
                scene_id=np.asarray(scene_id),
                height=np.asarray(H),
                width=np.asarray(W),
            )

        pos = int(y.sum())
        rows.append(
            {
                "chip_id": chip_id,
                "mineral": mineral,
                "mine_name": mine_safe,
                "scene_id": scene_id,
                "npz_path": str(out_npz.relative_to(pu.REPO_ROOT)),
                "height": H,
                "width": W,
                "n_bands": B,
                "n_positive_pixels": pos,
                "positive_fraction": round(pos / float(H * W), 6),
                "split": "unassigned",
            }
        )
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"[dataset] wrote {n_ok} samples", file=sys.stderr)

    if not rows:
        raise RuntimeError(
            "No samples were built. Check that step 03 produced chips and step 04 produced masks."
        )

    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(
        f"[dataset] DONE: {n_ok} samples "
        f"(skipped: {n_skip_size} wrong-size, {n_skip_mask} no-mask) -> {out_dir}",
        file=sys.stderr,
    )
    return n_ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a per-sample .npz + manifest.csv ML-ready dataset."
    )
    ap.add_argument("--mineral", required=True)
    ap.add_argument(
        "--chip-size",
        type=int,
        default=128,
        help="Required chip side length in pixels. Use --any-size to skip the filter.",
    )
    ap.add_argument(
        "--any-size",
        action="store_true",
        help="Accept chips of any size (overrides --chip-size).",
    )
    ap.add_argument("--out-dir", default=None, help="Override dataset output dir.")
    ap.add_argument("--overwrite", action="store_true", help="Re-write existing .npz files.")
    args = ap.parse_args()

    chip_size = None if args.any_size else int(args.chip_size)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (pu.mineral_dir(args.mineral) / "dataset")
    )
    n = build(
        mineral=args.mineral,
        chip_size=chip_size,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )
    print(f"{n} samples written to {out_dir}")


if __name__ == "__main__":
    main()
