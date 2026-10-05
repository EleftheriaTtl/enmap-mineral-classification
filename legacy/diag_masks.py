#!/usr/bin/env python3
"""
diag_masks.py
=============

Answers one question: when a chip comes out fully masked, which quality layer
did it? Run it on chips already on disk.

    python diag_masks.py                     # summary over data/chips
    python diag_masks.py --inspect 3         # + re-read the COGs for 3 bad chips
"""
from __future__ import annotations

import argparse
import glob
from collections import Counter

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="data/chips")
    ap.add_argument("--inspect", type=int, default=3,
                    help="re-open the COGs for this many fully-masked chips")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.chips}/*/*.npz"))
    if not files:
        raise SystemExit(f"no chips under {args.chips}")

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        rows.append((f, float(d["valid"].mean()), float(d["patch_cloud"])))
    v = np.array([r[1] for r in rows])
    c = np.array([r[2] for r in rows])

    print(f"{len(rows):,} chips")
    print(f"  fully masked (valid == 0) : {(v == 0).sum():,}  ({100*(v==0).mean():.1f}%)")
    print(f"  mostly masked (valid <.5) : {(v < 0.5).sum():,}  ({100*(v<0.5).mean():.1f}%)")
    print(f"  clean       (valid > .95) : {(v > 0.95).sum():,}  ({100*(v>0.95).mean():.1f}%)")
    print(f"  valid pct   [0,5,25,50,75,100] : "
          f"{np.round(np.percentile(v, [0,5,25,50,75,100])*100, 1).tolist()}")
    print(f"  fully masked AND cloud == 0     : {((v == 0) & (c == 0)).sum():,}"
          "   <- not cloud's fault")
    print(f"  patch_cloud > 0                 : {(c > 0).sum():,}")
    per_scene = Counter(f.split("/")[-2][:44] for f, vv, _ in rows if vv == 0)
    if per_scene:
        print("  fully-masked chips by scene:")
        for s, n in per_scene.most_common(10):
            print(f"      {n:6,}  {s}")

    if args.inspect <= 0:
        return

    # ---- re-read the quality COGs, one layer at a time ---------------------
    import os
    import rasterio
    from rasterio.windows import Window

    import pipeline_utils as pu
    from build_chips_cog import QUALITY_ASSETS, scene_urls, transform_point, with_auth

    pu.load_dotenv()
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.XML")

    bad_files = [f for f, vv, _ in rows if vv == 0][: args.inspect]
    if not bad_files:
        print("\nno fully-masked chips to inspect")
        return

    print("\nper-layer breakdown (fraction of the 64x64 patch flagged, and the "
          "values present):")
    for f in bad_files:
        d = np.load(f, allow_pickle=False)
        scene = str(d["scene_id"])
        px = int(d["valid"].shape[0])
        lat, lon = float(d["lat"]), float(d["lon"])
        print(f"\n  {f}")
        print(f"  site {str(d['site_id'])}  {lat:.5f},{lon:.5f}  {scene[:44]}")
        urls = scene_urls(scene)
        with rasterio.open("/vsicurl/" + with_auth(urls["image"])) as src:
            row, col = src.index(*transform_point(src.crs, lon, lat))
            half = px // 2
            win = Window(int(col) - half, int(row) - half, px, px)
            nd = src.nodatavals[0]
            b = src.read(1, window=win)
            print(f"      SPECTRAL band1   nodata={nd}  "
                  f"frac==nodata {float((b == nd).mean()):.2f}  "
                  f"min {b.min()} max {b.max()}")
        for a in QUALITY_ASSETS:
            try:
                with rasterio.open("/vsicurl/" + with_auth(urls[a])) as q:
                    m = q.read(1, window=win)
            except Exception as e:
                print(f"      {a:24s} unavailable ({type(e).__name__})")
                continue
            vals = dict(zip(*[x.tolist() for x in np.unique(m, return_counts=True)]))
            print(f"      {a:24s} flagged {float((m != 0).mean()):6.2f}   values {vals}")


if __name__ == "__main__":
    main()
