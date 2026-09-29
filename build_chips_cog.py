#!/usr/bin/env python3
"""
build_chips_cog.py
==================

Builds the chip dataset by reading EnMAP Cloud-Optimized GeoTIFFs over HTTP
range requests. No archives are downloaded and nothing but the chips touches
disk.

Why this works:
  * every asset URL is derivable from the scene id, so the STAC catalogue is
    never contacted (useful when it is blocked, and faster when it is not)
  * the COGs are 512x512 pixel-interleaved tiles: one tile carries all 224
    bands, so the first chip in a tile costs ~50 MB and every other chip in
    that tile is free while GDAL caches it -> process a scene in one pass
  * the quality layers are separate COGs, so cloud can be measured over the
    3.8 km patch instead of the 30 km scene

Per chip it writes X (int16, SWIR bands), the valid mask, wavelengths, and the
patch-level cloud fraction.

Usage
-----
    python build_chips_cog.py --limit 5 --verbose      # try a few scenes
    python build_chips_cog.py --shard 0/4              # a quarter of the list
    python build_chips_cog.py --max-patch-cloud 0.2    # skip clouded patches

Inputs: scene_priority.csv, scene_groups.csv, sites_v2.csv (as before)
Output: data/chips/<scene_id>/<site_id>.npz + manifest
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE = "https://download.geoservice.dlr.de/ENMAP/files/L2A"
NODATA = -9999

# quality layers used to mark a pixel unusable
QUALITY_ASSETS = ("QL_QUALITY_CLOUD", "QL_QUALITY_CLOUDSHADOW", "QL_QUALITY_HAZE",
                  "QL_QUALITY_CIRRUS", "QL_QUALITY_SNOW")


def _log(msg: str) -> None:
    print(f"[cog {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# URLs straight from the scene id
# ---------------------------------------------------------------------------
SCENE_RE = re.compile(r"^(ENMAP01-____L2A-(DT\d+)_(\d{4})(\d{2})(\d{2})T\d{6}Z_(\d{3})_.*)$")


def scene_urls(scene_id: str) -> dict[str, str]:
    """
    ENMAP01-____L2A-DT0000124671_20250416T003107Z_003_V010502_20250418T141036Z
      -> .../L2A/2025/04/16/DT0000124671/03/<scene_id>/<scene_id>-<ASSET>
    """
    m = SCENE_RE.match(scene_id)
    if not m:
        raise ValueError(f"unrecognised scene id: {scene_id}")
    full, dt, yyyy, mm, dd, counter = m.groups()
    folder = f"{BASE}/{yyyy}/{mm}/{dd}/{dt}/{int(counter):02d}/{full}/{full}"
    out = {"metadata": f"{folder}-METADATA.XML",
           "image": f"{folder}-SPECTRAL_IMAGE_COG.TIF"}
    for a in QUALITY_ASSETS + ("QL_PIXELMASK",):
        out[a] = f"{folder}-{a}_COG.TIF"
    return out


def with_auth(url: str) -> str:
    u = os.environ.get("ENMAP_USERNAME", "")
    p = os.environ.get("ENMAP_PASSWORD", "")
    return url.replace("https://", f"https://{u}:{p}@") if u and p else url


# ---------------------------------------------------------------------------
# wavelengths / SWIR band selection, from the scene's own metadata
# ---------------------------------------------------------------------------
def band_plan(meta_url: str, version: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (band_indices 1-based for rasterio, wavelengths) for the SWIR set."""
    import httpx

    with httpx.Client(timeout=120, follow_redirects=True) as c:
        r = c.get(meta_url, auth=(os.environ.get("ENMAP_USERNAME", ""),
                                  os.environ.get("ENMAP_PASSWORD", "")))
        r.raise_for_status()
        xml = r.content

    root = ET.fromstring(xml)
    strip = lambda t: t.split("}")[-1]

    centres: list[float] = []
    for band in root.iter():
        if strip(band.tag) != "bandID":
            continue
        for child in band:
            if strip(child.tag) == "wavelengthCenterOfBand" and child.text:
                centres.append(float(child.text))
                break
    wl = np.asarray(centres, dtype=np.float32)
    if wl.size == 0:
        raise ValueError("no wavelengths in metadata")

    swir: list[float] = []
    for el in root.iter():
        if strip(el.tag) != "SWIR":
            continue
        for sub in el.iter():
            if strip(sub.tag) == "wavelength" and sub.text:
                swir.append(float(sub.text))
    if swir:
        sw = np.asarray(swir, dtype=np.float32)
        keep = np.array(sorted({int(np.argmin(np.abs(wl - w))) for w in sw}), dtype=int)
    else:                                   # fall back: everything past 900 nm
        keep = np.where(wl >= 900.0)[0]

    if version == "128":                    # drop the 1330-1450 nm water band
        keep = keep[~((wl[keep] > 1330.86) & (wl[keep] < 1449.43))]
    elif version == "58":
        keep = keep[wl[keep] >= 1967.66]

    order = np.argsort(wl[keep])
    keep = keep[order]
    return keep + 1, wl[keep]               # rasterio bands are 1-based


# ---------------------------------------------------------------------------
def process_scene(scene_id: str, sites: pd.DataFrame, *, out_dir: Path, chip_px: int,
                  version: str, max_patch_cloud: float, verbose: bool) -> tuple[int, int, int, str]:
    """Returns (written, skipped_edge, skipped_cloud, error)."""
    import rasterio
    from rasterio.windows import Window

    urls = scene_urls(scene_id)
    scene_dir = out_dir / scene_id
    written = skip_edge = skip_cloud = 0

    bands, wl = band_plan(with_auth(urls["metadata"]), version)

    with rasterio.open("/vsicurl/" + with_auth(urls["image"])) as src:
        quality = {}
        for a in QUALITY_ASSETS:
            try:
                quality[a] = rasterio.open("/vsicurl/" + with_auth(urls[a]))
            except Exception:
                pass                        # a missing layer is not fatal
        try:
            half = chip_px // 2
            for _, s in sites.iterrows():
                out = scene_dir / f"{s.site_id}.npz"
                if out.exists():
                    continue
                row, col = src.index(*transform_point(src.crs, float(s.lon), float(s.lat)))
                r0, c0 = int(row) - half, int(col) - half
                if r0 < 0 or c0 < 0 or r0 + chip_px > src.height or c0 + chip_px > src.width:
                    skip_edge += 1
                    continue
                win = Window(c0, r0, chip_px, chip_px)

                bad = np.zeros((chip_px, chip_px), dtype=bool)
                cloud_frac = 0.0
                for a, q in quality.items():
                    m = q.read(1, window=win) != 0
                    if a == "QL_QUALITY_CLOUD":
                        cloud_frac = float(m.mean())
                    bad |= m
                if cloud_frac > max_patch_cloud:
                    skip_cloud += 1
                    if verbose:
                        _log(f"      {s.site_id}: patch cloud {cloud_frac:.0%}, skipped")
                    continue

                X = src.read(bands.tolist(), window=win).astype(np.int16)
                X[:, bad] = NODATA
                valid = (~bad).astype(np.uint8)

                scene_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out, X=X, valid=valid, wavelength=wl.astype(np.float32),
                    site_id=str(s.site_id), lat=float(s.lat), lon=float(s.lon),
                    scene_id=scene_id, minerals=str(s.get("geomaterials", "")),
                    patch_cloud=np.float32(cloud_frac),
                    scale_factor=np.float32(1e-4), nodata=np.int16(NODATA),
                )
                written += 1
                if verbose:
                    _log(f"      {s.site_id}: {X.shape}, cloud {cloud_frac:.0%}, "
                         f"valid {valid.mean():.0%}")
        finally:
            for q in quality.values():
                q.close()
    return written, skip_edge, skip_cloud, ""


_TX: dict = {}


def transform_point(crs, lon: float, lat: float) -> tuple[float, float]:
    from pyproj import Transformer
    key = str(crs)
    if key not in _TX:
        _TX[key] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return _TX[key].transform(lon, lat)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", default="data/coverage_v2_c50q2/scene_priority.csv")
    ap.add_argument("--scene-groups", default="data/coverage_v2_c50q2/scene_groups.csv")
    ap.add_argument("--sites", default="data/sites_v2.csv")
    ap.add_argument("--out-dir", default="data/chips")
    ap.add_argument("--chip-px", type=int, default=64)
    ap.add_argument("--version", default="128", choices=["v1", "128", "58"])
    ap.add_argument("--max-patch-cloud", type=float, default=0.2,
                    help="skip a chip when this fraction of its pixels is cloud")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--max-gb", type=float, default=130.0)
    ap.add_argument("--cache-mb", type=int, default=2048, help="GDAL block cache")
    ap.add_argument("--max-net-failures", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import pipeline_utils as pu
    pu.load_dotenv()
    if not (os.environ.get("ENMAP_USERNAME") and os.environ.get("ENMAP_PASSWORD")):
        sys.exit("ENMAP_USERNAME / ENMAP_PASSWORD not set in .env")

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.XML")
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
    os.environ.setdefault("GDAL_HTTP_VERSION", "2")
    os.environ["GDAL_CACHEMAX"] = str(args.cache_mb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    tag = ("_" + args.shard.replace("/", "of")) if args.shard else ""
    manifest_path = out_dir / f"manifest{tag}.csv"
    done: set[str] = set()
    if manifest_path.exists():
        m = pd.read_csv(manifest_path)
        done = set(m.loc[m.status == "ok", "scene_id"].astype(str))
        _log(f"resuming: {len(done):,} scenes already built")

    rows, consec_net = [], 0
    t0 = time.monotonic()
    n_chips_total = 0
    for k, sid in enumerate(scenes, 1):
        if sid in done:
            continue
        gb = sum(f.stat().st_size for f in out_dir.rglob("*.npz")) / 1e9
        if gb >= args.max_gb:
            _log(f"dataset reached {gb:.1f} GB (--max-gb); stopping")
            break

        ids = [x for x in str(grp.get(sid, "")).split(";") if x]
        want = by_id.reindex([i for i in ids if i in by_id.index]).reset_index()
        if want.empty:
            rows.append({"scene_id": sid, "status": "no_sites", "chips": 0, "error": ""})
            continue

        t = time.monotonic()
        _log(f"{k}/{len(scenes)} {sid[:44]}  {len(want)} chips  [{gb:.1f} GB]")
        try:
            w, se, sc, err = process_scene(sid, want, out_dir=out_dir, chip_px=args.chip_px,
                                           version=args.version,
                                           max_patch_cloud=args.max_patch_cloud,
                                           verbose=args.verbose)
            n_chips_total += w
            rows.append({"scene_id": sid, "status": "ok" if not err else "error",
                         "chips": w, "edge": se, "cloudy": sc, "error": err})
            consec_net = 0
            _log(f"      {w} written, {se} off-edge, {sc} too cloudy, "
                 f"{time.monotonic() - t:.0f}s")
        except KeyboardInterrupt:
            break
        except Exception as e:
            name = type(e).__name__
            rows.append({"scene_id": sid, "status": "error", "chips": 0, "error": repr(e)[:300]})
            if any(w in name for w in ("Timeout", "Connect", "Network", "CPLE", "RasterioIO")):
                consec_net += 1
                _log(f"      network/IO error ({name}); {consec_net} in a row")
                if consec_net >= args.max_net_failures:
                    _log("stopping: the download host is unreachable. Progress is saved.")
                    break
            else:
                _log(f"      ERROR {e!r}")
                traceback.print_exc()
        if len(rows) >= 5:
            pd.DataFrame(rows).to_csv(manifest_path, mode="a",
                                      header=not manifest_path.exists(), index=False)
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(manifest_path, mode="a",
                                  header=not manifest_path.exists(), index=False)
    gb = sum(f.stat().st_size for f in out_dir.rglob("*.npz")) / 1e9
    n = sum(1 for _ in out_dir.rglob("*.npz"))
    mins = (time.monotonic() - t0) / 60
    _log(f"done: {n:,} chips on disk ({gb:.1f} GB), {n_chips_total:,} written this run, "
         f"{mins:.0f} min" + (f", {n_chips_total / mins:.0f} chips/min" if mins else ""))


if __name__ == "__main__":
    main()
