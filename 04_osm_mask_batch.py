#!/usr/bin/env python3
"""
04_osm_mask_batch.py
====================

Step 4 of the EnMAP mineral-classification pipeline.

For every chip NetCDF under `data/<mineral>/chips/<mine>/*.nc`, query
OpenStreetMap for mine/quarry features inside the chip's bounding box and
rasterize them onto the chip's grid. Mask is saved as
`data/<mineral>/masks/<mine>/<chip_stem>_osm_mask.npy` (uint8, 0/1).

OSM tags used:

    landuse   = quarry, mine
    industrial = mining
    man_made  = tailings_pond, tailings, spoil_heap

Usage
-----
    python 04_osm_mask_batch.py --mineral magnesite
    python 04_osm_mask_batch.py --mineral magnesite --all-touched

Legacy mode (writes mask next to chip, like the old script):

    python 04_osm_mask_batch.py --legacy-out-dir "/Volumes/T7 Shield/sphalerite_output"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from shapely.geometry import box
from pyproj import CRS, Transformer
import osmnx as ox
from rasterio.features import rasterize

import pipeline_utils as pu


DEFAULT_TAGS = {
    "landuse": ["quarry", "mine"],
    "industrial": ["mining"],
    "man_made": ["tailings_pond", "tailings", "spoil_heap"],
}


def load_crs_from_wkt(ds: xr.Dataset) -> CRS:
    wkt = ds.attrs.get("crs_wkt")
    if wkt is None and "spatial_ref" in ds.variables:
        wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
    if wkt is None:
        raise ValueError("No CRS WKT found in dataset (attrs['crs_wkt'] or spatial_ref).")
    return CRS.from_wkt(wkt)


def mask_for_chip(nc_path: Path, *, tags: dict, all_touched: bool = False) -> np.ndarray:
    ds = xr.open_dataset(nc_path, engine="netcdf4", mask_and_scale=False)
    crs = load_crs_from_wkt(ds)

    template = ds["reflectance"].isel(wavelength=0)
    template = template.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    transform = template.rio.transform()
    height, width = template.sizes["y"], template.sizes["x"]

    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    xmin, xmax = sorted([left, right])
    ymin, ymax = sorted([bottom, top])

    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minlon, minlat = to_ll.transform(xmin, ymin)
    maxlon, maxlat = to_ll.transform(xmax, ymax)
    minlon, maxlon = sorted([minlon, maxlon])
    minlat, maxlat = sorted([minlat, maxlat])

    poly_ll = box(minlon, minlat, maxlon, maxlat)

    gdf = ox.features_from_polygon(poly_ll, tags)
    gdf = gdf[gdf.geometry.notnull()].copy()
    if len(gdf) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(gdf) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    gdf = gdf.to_crs(crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OSM mine/quarry masks for chip NetCDFs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mineral", help="Mineral name (uses data/<mineral>/chips and writes data/<mineral>/masks).")
    g.add_argument(
        "--legacy-out-dir",
        help="Legacy mode: writes mask next to each chip (mirrors the old 03_osm_mask_batch.py).",
    )
    ap.add_argument("--pattern", default="**/*.nc", help="Glob pattern under chips dir.")
    ap.add_argument("--all-touched", action="store_true", help="Use all_touched=True for fatter masks.")
    args = ap.parse_args()

    if args.mineral:
        chips_root = pu.chips_dir(args.mineral)
        masks_root: Path | None = pu.masks_dir(args.mineral)
    else:
        chips_root = Path(args.legacy_out_dir)
        masks_root = None  # write next to chip

    nc_files = [p for p in chips_root.glob(args.pattern) if p.is_file() and p.suffix.lower() == ".nc"]
    nc_files = sorted(nc_files)
    if not nc_files:
        print(f"[mask] no chip .nc files under {chips_root}", file=sys.stderr)
        sys.exit(1)

    for nc in nc_files:
        try:
            mask = mask_for_chip(nc, tags=DEFAULT_TAGS, all_touched=args.all_touched)
        except Exception as e:
            print(f"[mask] FAIL {nc}: {e}", file=sys.stderr)
            continue

        if masks_root is None:
            out_mask = Path(nc.with_suffix("").as_posix() + "_osm_mask.npy")
        else:
            mine_name = nc.parent.name  # chips/<mine>/<chip>.nc
            out_dir = masks_root / mine_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_mask = out_dir / f"{nc.stem}_osm_mask.npy"

        np.save(out_mask, mask.astype(np.uint8))
        print(f"[mask] OK {nc.name} sum={int(mask.sum())} -> {out_mask}")


if __name__ == "__main__":
    main()
