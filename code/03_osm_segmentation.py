#!/usr/bin/env python3
"""
03_osm_segmentation.py
======================

Step 3 of 3, optional. Turns the classification dataset into a segmentation
dataset: keep only the chips where OpenStreetMap actually maps mining ground
(an open pit, a quarry, a tailings pond, a shaft) and rasterize those polygons
onto the chip's own pixel grid as the target mask.

    stage query   group the chips into --cell-deg cells and run one Overpass
                  query per cell, caching the features in
                  data/interim/osm/<cell>.geojson. Resumable: a cached cell is
                  never queried twice.
    stage masks   rasterize the cached features onto each chip's grid, using
                  the CRS and affine transform stored in the chip itself, so
                  the mask lines up pixel for pixel with X.
    stage index   write data/seg/index.csv over the chips that have a non-empty
                  mask, which is the segmentation subset.

Queries go per cell rather than per chip because Overpass is the bottleneck:
one 0.25-degree cell covers ~27 km and usually several chips.

Usage
-----
    python code/03_osm_segmentation.py --stage query --limit 50   # try it
    python code/03_osm_segmentation.py --stage query
    python code/03_osm_segmentation.py --stage masks
    python code/03_osm_segmentation.py --stage index

Outputs
    data/interim/osm/<cell>.geojson     cached OSM features per cell
    data/seg/<site_id>.npz              mask uint8 (px,px) + the chip it belongs to
    data/seg/index.csv                  one row per kept chip
    data/reports/03_seg.{json,md}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TAG = "osm"
OSM_DIR = C.INTERIM / "osm"

# What OSM calls mining ground. landuse=quarry carries most of it worldwide;
# the rest is for shafts, adits and waste that sit outside the pit outline.
TAGS = {
    "landuse": ["quarry", "mine"],
    "industrial": ["mine", "mining", "quarry"],
    "man_made": ["mineshaft", "adit", "tailings_pond", "tailings", "spoil_heap"],
}


def cell_name(lat: float, lon: float, deg: float) -> str:
    return f"{np.floor(lat / deg) * deg:+08.3f}_{np.floor(lon / deg) * deg:+09.3f}"


def cell_bounds(name: str, deg: float) -> tuple[float, float, float, float]:
    la, lo = name.split("_")
    lat, lon = float(la), float(lo)
    return lon, lat, lon + deg, lat + deg


def read_index() -> pd.DataFrame:
    if not C.CHIP_INDEX_CSV.exists():
        sys.exit(f"{C.CHIP_INDEX_CSV} not found; run step 02 --stage index first")
    idx = pd.read_csv(C.CHIP_INDEX_CSV, low_memory=False)
    idx["site_id"] = idx.site_id.astype(str)
    return idx


# ==========================================================================
# stage: query
# ==========================================================================
def stage_query(args) -> None:
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import box

    OSM_DIR.mkdir(parents=True, exist_ok=True)
    idx = read_index()
    idx["cell"] = [cell_name(la, lo, args.cell_deg)
                   for la, lo in zip(idx.lat, idx.lon)]
    cells = (idx.groupby("cell").size().sort_values(ascending=False)
                .reset_index(name="chips"))
    todo = [c for c in cells.cell if not (OSM_DIR / f"{c}.geojson").exists()]
    C.log(TAG, f"{len(idx):,} chips in {len(cells):,} cells of {args.cell_deg} deg; "
               f"{len(todo):,} cells still to query")
    if args.limit:
        todo = todo[: args.limit]

    t0, empty, failed = time.monotonic(), 0, 0
    for i, cell in enumerate(todo, 1):
        poly = box(*cell_bounds(cell, args.cell_deg))
        out = OSM_DIR / f"{cell}.geojson"
        gdf = None
        for attempt in range(1, args.retries + 1):
            try:
                gdf = ox.features_from_polygon(poly, TAGS)
                break
            except Exception as e:
                # osmnx raises rather than returning empty when nothing matches
                if ("InsufficientResponse" in type(e).__name__
                        or "No matching features" in str(e)):
                    gdf = None
                    break
                if attempt == args.retries:
                    C.log(TAG, f"  {cell}: giving up ({type(e).__name__})")
                    failed += 1
                    gdf = False                   # distinguish failure from empty
                else:
                    time.sleep(args.backoff * attempt)
        if gdf is False:
            continue                              # retry on a later run
        if gdf is None or len(gdf) == 0:
            # write an empty file so the cell is not queried again
            gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326").to_file(out, driver="GeoJSON")
            empty += 1
        else:
            keep = gdf[gdf.geometry.notnull()].copy()
            keep = keep[["geometry"]].reset_index(drop=True)
            keep.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
        if args.sleep:
            time.sleep(args.sleep)
        if i % 25 == 0 or i == len(todo):
            rate = i / max(1e-6, time.monotonic() - t0)
            C.log(TAG, f"  {i}/{len(todo)} cells ({rate * 60:.0f}/min, "
                       f"{empty} empty, {failed} failed), ETA "
                       f"{(len(todo) - i) / max(1e-6, rate) / 60:.0f} min")
    C.log(TAG, f"query done: {len(todo) - failed:,} cells cached, {failed:,} to retry")


# ==========================================================================
# stage: masks
# ==========================================================================
def stage_masks(args) -> None:
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import rasterize
    from shapely.geometry import box

    C.SEG.mkdir(parents=True, exist_ok=True)
    idx = read_index()
    idx["cell"] = [cell_name(la, lo, args.cell_deg)
                   for la, lo in zip(idx.lat, idx.lon)]

    rows, kept, skipped_nocache = [], 0, 0
    for cell, grp in idx.groupby("cell"):
        cache = OSM_DIR / f"{cell}.geojson"
        if not cache.exists():
            skipped_nocache += len(grp)
            continue
        feats = gpd.read_file(cache)
        if len(feats) == 0:
            continue
        for r in grp.itertuples():
            out = C.SEG / f"{r.site_id}.npz"
            if out.exists() and not args.force:
                continue
            chip = C.DATA / r.path
            if not chip.exists():
                continue
            d = np.load(chip, allow_pickle=False)
            if "transform" not in d or "crs_wkt" not in d:
                C.log(TAG, f"  {r.site_id}: chip has no grid stored; rebuild it "
                           f"with the current step 02")
                continue
            tr = Affine(*[float(x) for x in d["transform"]])
            crs = str(d["crs_wkt"])
            px = int(d["chip_px"]) if "chip_px" in d else int(d["valid"].shape[0])

            # chip footprint in lon/lat, to pre-select features
            xs = [tr.c, tr.c + tr.a * px]
            ys = [tr.f, tr.f + tr.e * px]
            from pyproj import Transformer
            to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lons, lats = to_ll.transform([min(xs), max(xs)], [min(ys), max(ys)])
            chip_ll = box(min(lons), min(lats), max(lons), max(lats))
            sub = feats[feats.intersects(chip_ll)]
            if len(sub) == 0:
                continue

            sub = sub.to_crs(crs)
            shapes = []
            for g in sub.geometry:
                if g is None or g.is_empty:
                    continue
                if g.geom_type in ("Point", "LineString", "MultiPoint", "MultiLineString"):
                    g = g.buffer(args.point_buffer_m)   # a shaft is a node in OSM
                shapes.append((g, 1))
            if not shapes:
                continue
            mask = rasterize(shapes, out_shape=(px, px), transform=tr, fill=0,
                             dtype="uint8", all_touched=args.all_touched)
            frac = float(mask.mean())
            if mask.sum() == 0:
                continue
            np.savez_compressed(out, mask=mask, site_id=str(r.site_id),
                                chip_path=str(r.path), scene_id=str(r.scene_id),
                                lat=float(r.lat), lon=float(r.lon),
                                minerals=str(r.minerals),
                                transform=np.array(list(tr)[:6], dtype=np.float64),
                                crs_wkt=crs, n_features=np.int32(len(shapes)),
                                mask_frac=np.float32(frac))
            rows.append({"site_id": r.site_id, "mask_frac": frac,
                         "n_features": len(shapes)})
            kept += 1
    C.log(TAG, f"{kept:,} chips got a non-empty mask")
    if skipped_nocache:
        C.log(TAG, f"{skipped_nocache:,} chips skipped: their cell is not cached yet "
                   f"(run --stage query)")


# ==========================================================================
# stage: index
# ==========================================================================
def stage_index(args) -> None:
    files = sorted(C.SEG.glob("*.npz"))
    if not files:
        sys.exit(f"no masks under {C.SEG}; run --stage masks first")
    chips = read_index().set_index("site_id")

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        sid = str(d["site_id"])
        base = chips.loc[sid] if sid in chips.index else None
        rows.append({
            "site_id": sid,
            "mask_path": str(f.relative_to(C.DATA)),
            "chip_path": str(d["chip_path"]),
            "scene_id": str(d["scene_id"]),
            "lat": float(d["lat"]), "lon": float(d["lon"]),
            "minerals": str(d["minerals"]),
            "mask_frac": float(d["mask_frac"]),
            "n_features": int(d["n_features"]),
            "patch_cloud": float(base.patch_cloud) if base is not None else None,
            "valid_frac": float(base.valid_frac) if base is not None else None,
        })
    seg = pd.DataFrame(rows).sort_values("mask_frac", ascending=False).reset_index(drop=True)
    seg.to_csv(C.SEG_INDEX_CSV, index=False)
    C.log(TAG, f"wrote {C.SEG_INDEX_CSV.name} ({len(seg):,} chips)")

    n_chips = len(chips)
    stats = {
        "chips_in": int(n_chips), "chips_with_mask": int(len(seg)),
        "share": float(len(seg) / max(1, n_chips)),
        "mask_frac": {"mean": float(seg.mask_frac.mean()),
                      "median": float(seg.mask_frac.median()),
                      "p95": float(seg.mask_frac.quantile(0.95))},
        "cells_cached": int(len(list(OSM_DIR.glob("*.geojson")))) if OSM_DIR.exists() else 0,
        "tags": TAGS,
    }
    md = ["# Step 3 — OSM segmentation subset", "",
          f"**{len(seg):,} of {n_chips:,} chips** ({100 * len(seg) / max(1, n_chips):.1f}%) "
          f"contain mining ground that OpenStreetMap has mapped. Those get a "
          f"pixel mask; the rest are classification-only.", "",
          "On average the mask covers "
          f"{stats['mask_frac']['mean']:.1%} of a chip "
          f"(median {stats['mask_frac']['median']:.1%}).", "",
          "## Tags treated as mining ground", ""]
    md += C.md_table(["Key", "Values"],
                     [(k, ", ".join(v)) for k, v in TAGS.items()])
    md += ["", "Nodes and ways (a shaft, an adit) are buffered to one pixel; "
               "areas are rasterized as they are.",
           "", f"Index: `{C.SEG_INDEX_CSV.relative_to(C.REPO)}`. A mask lines up "
               f"pixel for pixel with its chip's X, because it was rasterized "
               f"with the transform stored in the chip.",
           "", "Absence of a mask means OSM has not mapped the pit, not that "
               "there is no pit — treat the masks as positive-only labels."]
    C.write_report("03_seg", stats, "\n".join(md) + "\n")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "query", "masks", "index"])
    ap.add_argument("--cell-deg", type=float, default=0.25,
                    help="one Overpass query per cell of this size")
    ap.add_argument("--point-buffer-m", type=float, default=30.0,
                    help="radius given to an OSM node or way, in metres")
    ap.add_argument("--all-touched", action="store_true",
                    help="rasterize any pixel the polygon touches, not just centres")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="pause between Overpass queries; be polite")
    ap.add_argument("--limit", type=int, default=None, help="stop after N cells")
    ap.add_argument("--force", action="store_true", help="rebuild existing masks")
    args = ap.parse_args()

    if args.stage in ("all", "query"):
        stage_query(args)
    if args.stage in ("all", "masks"):
        stage_masks(args)
    if args.stage in ("all", "index"):
        stage_index(args)


if __name__ == "__main__":
    main()
