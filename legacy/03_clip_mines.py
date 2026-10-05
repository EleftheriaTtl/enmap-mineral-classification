#!/usr/bin/env python3
"""
03_clip_mines.py
================

Step 3 of the EnMAP mineral-classification pipeline.

For every NetCDF in `data/<mineral>/netcdf/`, clip a square chip centered on
each mine (lat, lon) from `data/<mineral>/localities.csv` and save to
`data/<mineral>/chips/<mine>/`.

The chip can be defined two ways:

    --buffer-m 1500     (default)  geographic buffer in meters
    --chip-px  128                 fixed pixel chip (legacy behavior)

If both are given, --buffer-m wins.

For each (mine, scene) the script writes:

    <mine>/<mine>_<scene>.nc                # clipped NetCDF
    <mine>/<mine>_<scene>_rgb.png           # quicklook (2200/1650/1200 nm)
    <mine>/<mine>_<scene>_reflectance.npy   # raw reflectance values
    <mine>/<mine>_<scene>_wavelength.npy    # wavelength axis

Usage
-----
    python 03_clip_mines.py --mineral magnesite                  # 1500 m buffer
    python 03_clip_mines.py --mineral magnesite --buffer-m 3000
    python 03_clip_mines.py --mineral magnesite --chip-px 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers .rio accessor)
from pyproj import CRS, Transformer

import matplotlib.pyplot as plt

import pipeline_utils as pu


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def save_rgb_png(ds: xr.Dataset, out_png: Path, stretch_p=(2, 98), scale=1e-4) -> None:
    w = ds["wavelength"].values.astype(float)

    def nearest(target: float) -> int:
        return int(np.argmin(np.abs(w - target)))

    r_i, g_i, b_i = nearest(2200), nearest(1650), nearest(1200)
    R = ds["reflectance"].isel(wavelength=r_i).values * scale
    G = ds["reflectance"].isel(wavelength=g_i).values * scale
    B = ds["reflectance"].isel(wavelength=b_i).values * scale

    rgb = np.stack([R, G, B], axis=-1)
    lo = np.nanpercentile(rgb, stretch_p[0], axis=(0, 1))
    hi = np.nanpercentile(rgb, stretch_p[1], axis=(0, 1))
    rgb = (rgb - lo) / (hi - lo + 1e-12)
    rgb = np.clip(rgb, 0, 1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png.as_posix(), rgb)


def ensure_ds_crs(ds: xr.Dataset) -> tuple[xr.Dataset, CRS]:
    crs = getattr(ds.rio, "crs", None)
    if crs is None or not str(crs).strip():
        wkt = ds.attrs.get("crs_wkt") or (
            ds["reflectance"].attrs.get("crs_wkt") if "reflectance" in ds else None
        )
        if not wkt:
            raise ValueError("No CRS found (ds.rio.crs is None and no crs_wkt attr).")
        crs = CRS.from_wkt(wkt)
        ds = ds.rio.write_crs(crs)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return ds, crs


def _pixel_size(ds: xr.Dataset) -> float:
    """Mean absolute pixel size in CRS units (typically meters)."""
    xs = ds["x"].values
    ys = ds["y"].values
    dx = float(np.abs(np.diff(xs)).mean()) if xs.size > 1 else 30.0
    dy = float(np.abs(np.diff(ys)).mean()) if ys.size > 1 else 30.0
    return (dx + dy) / 2.0


def clip_chip_save(
    ds: xr.Dataset,
    *,
    lat: float,
    lon: float,
    chip_px: Optional[int],
    buffer_m: Optional[float],
    out_nc: Path,
    out_png: Path,
) -> Optional[xr.Dataset]:
    if (chip_px is None) == (buffer_m is None):
        raise ValueError("Exactly one of chip_px / buffer_m must be set.")

    ds, crs = ensure_ds_crs(ds)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = transformer.transform(lon, lat)

    minx, miny, maxx, maxy = ds.rio.bounds()
    if not (minx <= x0 <= maxx and miny <= y0 <= maxy):
        return None

    xs = ds["x"].values
    ys = ds["y"].values
    ix = int(np.abs(xs - x0).argmin())
    iy = int(np.abs(ys - y0).argmin())

    if buffer_m is not None:
        gsd = _pixel_size(ds)
        half = max(1, int(round(buffer_m / gsd)))
    else:
        if chip_px % 2 != 0:
            raise ValueError("chip_px must be even (e.g. 128).")
        half = chip_px // 2

    x0_i, x1_i = ix - half, ix + half
    y0_i, y1_i = iy - half, iy + half

    if x0_i < 0 or y0_i < 0 or x1_i > xs.shape[0] or y1_i > ys.shape[0]:
        return None

    clipped = ds.isel(x=slice(x0_i, x1_i), y=slice(y0_i, y1_i))
    if clipped.sizes["x"] == 0 or clipped.sizes["y"] == 0:
        return None

    clipped = clipped.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    clipped = clipped.rio.write_crs(crs, inplace=False)

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "reflectance": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (
                min(10, clipped.sizes["wavelength"]),
                min(256, clipped.sizes["y"]),
                min(256, clipped.sizes["x"]),
            ),
        }
    }
    clipped.to_netcdf(out_nc, engine="netcdf4", format="NETCDF4", encoding=encoding)

    save_rgb_png(clipped, out_png)

    ref = clipped["reflectance"].values.astype(np.float32)
    wl = clipped["wavelength"].values.astype(np.float32)
    np.save(out_nc.with_suffix("").as_posix() + "_reflectance.npy", ref)
    np.save(out_nc.with_suffix("").as_posix() + "_wavelength.npy", wl)

    return clipped


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Clip per-mine chips out of EnMAP NetCDFs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mineral", help="Mineral name; uses data/<mineral>/...")
    g.add_argument("--base-dir", help="Legacy mode: explicit base dir.")
    ap.add_argument("--mines-csv", default=None, help="Override localities CSV path.")
    ap.add_argument("--out-dir", default=None, help="Override chip output dir.")
    ap.add_argument("--netcdf-subdir", default="netcdf", help="Subdir of NetCDFs (legacy: 'netcdf_outputs').")
    ap.add_argument("--buffer-m", type=float, default=1500.0, help="Geographic buffer in meters (default 1500).")
    ap.add_argument("--chip-px", type=int, default=None, help="Override with fixed pixel chip size (even number).")
    args = ap.parse_args()

    if args.mineral:
        nc_dir = pu.netcdf_dir(args.mineral)
        chips_root = Path(args.out_dir) if args.out_dir else pu.chips_dir(args.mineral)
        mines_csv = Path(args.mines_csv) if args.mines_csv else pu.localities_csv(args.mineral)
    else:
        base_dir = Path(args.base_dir)
        nc_dir = base_dir / args.netcdf_subdir
        chips_root = Path(args.out_dir) if args.out_dir else (base_dir / "chips")
        if not args.mines_csv:
            print("--mines-csv is required when using --base-dir.", file=sys.stderr)
            sys.exit(2)
        mines_csv = Path(args.mines_csv)

    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files under {nc_dir}")

    df = pu.load_localities(str(mines_csv))

    # which sizing wins
    if args.chip_px is not None:
        chip_px: Optional[int] = args.chip_px
        buffer_m: Optional[float] = None
    else:
        chip_px = None
        buffer_m = args.buffer_m

    chips_root.mkdir(parents=True, exist_ok=True)

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file, engine="netcdf4", mask_and_scale=False)
        try:
            ds, crs = ensure_ds_crs(ds)
        except Exception as e:
            print(f"[clip] SKIP {nc_file.name}: {e}", file=sys.stderr)
            continue

        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        minx, miny, maxx, maxy = ds.rio.bounds()

        for _, row in df.iterrows():
            mine_name = pu.safe_name(row["Name"])
            lat, lon = float(row["lat"]), float(row["lon"])
            x, y = transformer.transform(lon, lat)
            if not (minx <= x <= maxx and miny <= y <= maxy):
                continue

            src = nc_file.stem
            mine_dir = chips_root / mine_name
            out_nc = mine_dir / f"{mine_name}_{src}.nc"
            out_png = mine_dir / f"{mine_name}_{src}_rgb.png"

            clipped = clip_chip_save(
                ds,
                lat=lat,
                lon=lon,
                chip_px=chip_px,
                buffer_m=buffer_m,
                out_nc=out_nc,
                out_png=out_png,
            )
            ok = clipped is not None
            print(f"[clip] {mine_name} @ {src}: saved={ok}")


if __name__ == "__main__":
    main()
