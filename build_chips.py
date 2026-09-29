#!/usr/bin/env python3
"""
build_chips.py
==============

Scene-driven dataset builder. Walks the greedy scene order, and for each scene:

    download archive -> read (SWIR, cloud-masked) -> cut every chip inside it
    -> write one .npz per chip as int16 -> delete the archive

Peak disk is one archive plus the growing dataset, not the total download volume.

Chips are int16, which is what EnMAP L2A actually ships: step 02 upcasts to
float32 and doubles the size for nothing. Masked pixels are stored as -9999 and
flagged in a `valid` mask, so no information is lost.

Usage
-----
    # try 10 scenes first
    python build_chips.py --limit 10

    # the real run, 4 shards in 4 tmux windows
    python build_chips.py --shard 0/4
    python build_chips.py --shard 1/4   # ...etc

    # options
    python build_chips.py --chip-px 64 --version 128 --max-gb 130

Inputs
    data/coverage_v2_c50q2/scene_priority.csv   scene order (greedy)
    data/coverage_v2_c50q2/scene_groups.csv     scene -> site ids
    data/sites_v2.csv                           site_id, lat, lon, geomaterials

Outputs
    data/chips/<scene_id>/<site_id>.npz         X int16 (B,H,W), valid uint8, wavelength
    data/chips/manifest.csv                     one row per scene: status, chips, error
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pipeline_utils as pu

NODATA = -9999


def _log(msg: str) -> None:
    print(f"[chips {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*.npz")) / 1e9


# ---------------------------------------------------------------------------
def cut_chip(ds, lat: float, lon: float, px: int):
    """Return (X int16 (B,px,px), valid uint8 (px,px)) centred on lat/lon, or None."""
    from pyproj import Transformer

    crs = ds.rio.crs if hasattr(ds, "rio") and ds.rio.crs else ds.attrs.get("crs")
    if crs is None:
        raise ValueError("dataset has no CRS")
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tx.transform(lon, lat)

    xs = ds["x"].values
    ys = ds["y"].values
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))

    half = px // 2
    x0, x1 = ix - half, ix + half
    y0, y1 = iy - half, iy + half
    if x0 < 0 or y0 < 0 or x1 > len(xs) or y1 > len(ys):
        return None                                  # window falls off the scene

    arr = ds["reflectance"].values[:, y0:y1, x0:x1]   # (B, px, px) float32, NaN = masked
    valid = np.isfinite(arr).all(axis=0).astype(np.uint8)
    X = np.where(np.isfinite(arr), np.rint(arr), NODATA)
    X = np.clip(X, -32768, 32767).astype(np.int16)    # L2A reflectance is int16 * 1e-4
    return X, valid


def process_scene(scene_id: str, sites: pd.DataFrame, *, out_dir: Path, raw_dir: Path,
                  chip_px: int, version: str, keep_zip: bool, user: str, pwd: str,
                  glog) -> tuple[int, int, str]:
    """Returns (chips_written, chips_skipped, error)."""
    import geoserver_download as gsd
    from new_enmap import EnMAPProcessor

    scene_dir = out_dir / scene_id
    zip_path = raw_dir / f"{scene_id}.zip"

    try:
        if not (zip_path.exists() and zip_path.stat().st_size > 0):
            item = fetch_item(scene_id)
            if not gsd.download_item(item, raw_dir, username=user, password=pwd, logger=glog):
                return 0, 0, "download failed (auth or empty archive)"

        proc = EnMAPProcessor(base_dir=str(raw_dir), save_netcdf=False, apply_cloud_mask=True,
                              keep_only_swir=True, version=version)
        ds = proc.read_enmap_zip(zip_path)
        if ds is None:
            return 0, 0, "processor returned no dataset"

        scene_dir.mkdir(parents=True, exist_ok=True)
        wl = ds["wavelength"].values.astype(np.float32)
        written = skipped = 0
        for _, s in sites.iterrows():
            out = scene_dir / f"{s.site_id}.npz"
            if out.exists():
                continue
            try:
                got = cut_chip(ds, float(s.lat), float(s.lon), chip_px)
            except Exception as e:
                _log(f"      chip {s.site_id}: {e!r}")
                got = None
            if got is None:
                skipped += 1
                continue
            X, valid = got
            np.savez_compressed(
                out, X=X, valid=valid, wavelength=wl,
                site_id=str(s.site_id), lat=float(s.lat), lon=float(s.lon),
                scene_id=scene_id, minerals=str(s.get("geomaterials", "")),
                scale_factor=np.float32(1e-4), nodata=np.int16(NODATA),
            )
            written += 1
        ds.close()
        del ds
        return written, skipped, ""
    finally:
        if not keep_zip:
            zip_path.unlink(missing_ok=True)


def fetch_item(scene_id: str) -> dict:
    import httpx
    import geoserver_download as gsd
    url = f"{gsd.STAC_BASE}/collections/{gsd.COLLECTION_ENMAP_L2A}/items/{scene_id}"
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        r = c.get(url, headers={"Accept": "application/geo+json"})
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", default="data/coverage_v2_c50q2/scene_priority.csv")
    ap.add_argument("--scene-groups", default="data/coverage_v2_c50q2/scene_groups.csv")
    ap.add_argument("--sites", default="data/sites_v2.csv")
    ap.add_argument("--out-dir", default="data/chips")
    ap.add_argument("--raw-dir", default="data/_raw_tmp")
    ap.add_argument("--chip-px", type=int, default=64)
    ap.add_argument("--version", default="128", choices=["v1", "128", "58"])
    ap.add_argument("--limit", type=int, default=None, help="stop after N scenes (for testing)")
    ap.add_argument("--shard", default=None, help="i/n: take every n-th scene, offset i")
    ap.add_argument("--max-gb", type=float, default=130.0, help="stop when the dataset reaches this")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    pu.load_dotenv()
    user = os.environ.get("ENMAP_USERNAME") or os.environ.get("EOWEB_USER")
    pwd = os.environ.get("ENMAP_PASSWORD") or os.environ.get("EOWEB_PASS")
    if not (user and pwd):
        sys.exit("ENMAP_USERNAME / ENMAP_PASSWORD not set in .env")

    out_dir, raw_dir = Path(args.out_dir), Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pri = pd.read_csv(args.priority)
    grp = pd.read_csv(args.scene_groups).set_index("scene_id")["site_ids"].to_dict()
    sites = pd.read_csv(args.sites, low_memory=False)
    sites["site_id"] = sites.site_id.astype(str)
    by_id = sites.set_index("site_id")

    scenes = pri.scene_id.tolist()
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        scenes = scenes[i::n]
        _log(f"shard {i}/{n}: {len(scenes):,} scenes")
    if args.limit:
        scenes = scenes[: args.limit]

    manifest_path = out_dir / f"manifest{('_' + args.shard.replace('/', 'of')) if args.shard else ''}.csv"
    done: dict[str, str] = {}
    if manifest_path.exists():
        m = pd.read_csv(manifest_path)
        done = dict(zip(m.scene_id.astype(str), m.status.astype(str)))
        _log(f"resuming: {sum(v == 'ok' for v in done.values()):,} scenes already built")

    import geoserver_download as gsd
    glog = gsd.get_logger(level=logging.WARNING)

    rows = []
    t0 = time.monotonic()
    for k, sid in enumerate(scenes, 1):
        if done.get(sid) == "ok":
            continue
        size = dir_size_gb(out_dir)
        if size >= args.max_gb:
            _log(f"dataset reached {size:.1f} GB (--max-gb {args.max_gb}); stopping")
            break

        ids = [x for x in str(grp.get(sid, "")).split(";") if x]
        want = by_id.reindex([i for i in ids if i in by_id.index]).reset_index()
        if want.empty:
            rows.append({"scene_id": sid, "status": "no_sites", "chips": 0, "error": ""})
            continue

        t = time.monotonic()
        _log(f"{k}/{len(scenes)} {sid[:44]}  {len(want)} chips  [{size:.1f} GB]")
        try:
            n_ok, n_skip, err = process_scene(
                sid, want, out_dir=out_dir, raw_dir=raw_dir, chip_px=args.chip_px,
                version=args.version, keep_zip=args.keep_zips, user=user, pwd=pwd, glog=glog)
            status = "ok" if not err else "error"
            rows.append({"scene_id": sid, "status": status, "chips": n_ok, "error": err})
            _log(f"      {n_ok} written, {n_skip} off-edge, {time.monotonic() - t:.0f}s"
                 + (f"  ERROR {err}" if err else ""))
        except KeyboardInterrupt:
            pd.DataFrame(rows).to_csv(manifest_path, mode="a", header=not manifest_path.exists(), index=False)
            raise
        except Exception as e:
            rows.append({"scene_id": sid, "status": "error", "chips": 0, "error": repr(e)[:300]})
            _log(f"      ERROR {e!r}")
            traceback.print_exc()
        if len(rows) >= 5:
            pd.DataFrame(rows).to_csv(manifest_path, mode="a", header=not manifest_path.exists(), index=False)
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(manifest_path, mode="a", header=not manifest_path.exists(), index=False)
    n_chips = sum(1 for _ in out_dir.rglob("*.npz"))
    _log(f"done: {n_chips:,} chips, {dir_size_gb(out_dir):.1f} GB, "
         f"{(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
