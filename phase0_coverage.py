#!/usr/bin/env python3
"""
phase0_coverage.py
==================

Phase 0, step 3: EnMAP L2A coverage for every site in sites.csv, without one
STAC request per site (171k sites would take ~95 hours).

Instead:

    harvest   page the whole ENMAP_HSI_L2A collection once (globe split into
              tiles, each tile paged) -> footprints.csv.gz
              columns: scene_id, datetime, cloud, quality, wkt

    join      point-in-polygon every site against the harvested footprints
              (shapely STRtree) -> site_coverage.csv, scene_groups.csv
              and a funnel report

    funnel    re-print the report from existing outputs (no recompute)

"Covered" is reported two ways:
    any     : site falls in >= 1 EnMAP L2A footprint
    usable  : >= 1 footprint with cloud <= --cloud-max and nominal quality

Usage
-----
    python phase0_coverage.py harvest --out data/footprints.csv.gz
    python phase0_coverage.py join --sites data/sites.csv \
        --footprints data/footprints.csv.gz --out-dir data/coverage
    python phase0_coverage.py funnel --out-dir data/coverage --localities 451221

Harvest is resumable: finished tiles are recorded in <out>.tiles.json and
skipped on a rerun. Run it in tmux.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

STAC_BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION = "ENMAP_HSI_L2A"
PAGE_LIMIT = 500

# measured on 2 scenes, 2026-09-22 (128 SWIR bands, 128x128 chips)
GB_RAW_PER_SCENE = 0.45
GB_NC_PER_SCENE = 0.21
MB_CHIP = 10.5
MB_SAMPLE = 6.0  # .npz, still an estimate


def _log(msg: str) -> None:
    print(f"[phase0 {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def _open(path: Path, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else open(path, mode, encoding="utf-8")


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------
def tiles(deg: float) -> List[Dict[str, Any]]:
    out = []
    lat = -90.0
    while lat < 90.0:
        top = min(lat + deg, 90.0)
        lon = -180.0
        while lon < 180.0:
            right = min(lon + deg, 180.0)
            out.append({"name": f"{lat:+06.1f}_{lon:+07.1f}", "bbox": [lon, lat, right, top]})
            lon = right
        lat = top
    return out


def page_tile(client, bbox, *, start: str, end: str, limit: int) -> Iterator[dict]:
    """Yield every item whose footprint intersects bbox, following 'next' links."""
    body: Dict[str, Any] = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "limit": limit,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
    }
    url = f"{STAC_BASE}/search"
    seen_tokens: set[str] = set()
    while True:
        r = client.post(url, json=body, headers={"Content-Type": "application/json",
                                                 "Accept": "application/geo+json"})
        r.raise_for_status()
        data = r.json()
        feats = data.get("features") or []
        for f in feats:
            yield f
        nxt = next((l for l in (data.get("links") or []) if l.get("rel") == "next"), None)
        if not nxt or not feats:
            return
        # STAC servers express 'next' either as a body merge (POST) or a href (GET)
        if nxt.get("body"):
            merged = {**body, **nxt["body"]} if nxt.get("merge", True) else dict(nxt["body"])
            token = json.dumps(merged, sort_keys=True)
            if token in seen_tokens:
                return
            seen_tokens.add(token)
            body, url = merged, nxt.get("href") or url
        elif nxt.get("href"):
            href = nxt["href"]
            if href in seen_tokens:
                return
            seen_tokens.add(href)
            rr = client.get(href, headers={"Accept": "application/geo+json"})
            rr.raise_for_status()
            data = rr.json()
            for f in data.get("features") or []:
                yield f
            # continue following hrefs from here
            while True:
                nxt2 = next((l for l in (data.get("links") or []) if l.get("rel") == "next"), None)
                if not nxt2 or not nxt2.get("href") or nxt2["href"] in seen_tokens:
                    return
                seen_tokens.add(nxt2["href"])
                rr = client.get(nxt2["href"], headers={"Accept": "application/geo+json"})
                rr.raise_for_status()
                data = rr.json()
                for f in data.get("features") or []:
                    yield f
        else:
            return


def props(f: dict) -> tuple[Optional[float], Optional[int], str]:
    p = f.get("properties") or {}
    cc = p.get("eo:cloud_cover", p.get("cloudCover", p.get("cloud_cover")))
    try:
        cc = float(cc) if cc is not None else None
    except (TypeError, ValueError):
        cc = None
    q = p.get("enmap:overallQuality", p.get("enmap:OverallQuality"))
    try:
        q = int(q) if q is not None else None
    except (TypeError, ValueError):
        q = None
    return cc, q, str(p.get("datetime") or "")


def geom_wkt(f: dict) -> Optional[str]:
    from shapely.geometry import shape
    g = f.get("geometry")
    if not g:
        bbox = f.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        from shapely.geometry import box
        return box(bbox[0], bbox[1], bbox[2], bbox[3]).wkt
    try:
        return shape(g).wkt
    except Exception:
        return None


def harvest(args) -> None:
    import httpx

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(out.suffix + ".tiles.json")
    done: set[str] = set()
    if state_path.exists() and not args.restart:
        done = set(json.loads(state_path.read_text()).get("done", []))
        _log(f"resuming: {len(done)} tiles already harvested")

    seen: set[str] = set()
    mode = "a" if (out.exists() and done and not args.restart) else "w"
    if mode == "a":
        with _open(out, "r") as fh:
            next(fh, None)
            for line in fh:
                seen.add(line.split(",", 1)[0])
        _log(f"resuming: {len(seen)} scenes already on disk")

    cells = tiles(args.tile_deg)
    todo = [c for c in cells if c["name"] not in done]
    _log(f"{len(cells)} tiles of {args.tile_deg} deg, {len(todo)} to go "
         f"({args.start}..{args.end})")

    t0 = time.monotonic()
    with _open(out, mode) as fh, httpx.Client(timeout=120) as client:
        if mode == "w":
            fh.write("scene_id,datetime,cloud,quality,wkt\n")
        for i, c in enumerate(todo, 1):
            got = 0
            for attempt in range(1, args.retries + 1):
                try:
                    for f in page_tile(client, c["bbox"], start=args.start, end=args.end,
                                       limit=args.limit):
                        sid = f.get("id")
                        if not sid or sid in seen:
                            continue
                        wkt = geom_wkt(f)
                        if not wkt:
                            continue
                        cc, q, dt = props(f)
                        seen.add(sid)
                        got += 1
                        fh.write(f'{sid},{dt},{"" if cc is None else cc},'
                                 f'{"" if q is None else q},"{wkt}"\n')
                    break
                except Exception as e:
                    if attempt == args.retries:
                        _log(f"  tile {c['name']}: giving up after {attempt} tries ({e!r})")
                    else:
                        time.sleep(2 * attempt)
            done.add(c["name"])
            fh.flush()
            state_path.write_text(json.dumps({"done": sorted(done)}))
            if i % 10 == 0 or i == len(todo):
                rate = i / max(1e-6, time.monotonic() - t0)
                _log(f"  {i}/{len(todo)} tiles, {len(seen)} scenes, "
                     f"ETA {(len(todo) - i) / max(1e-6, rate) / 60:.0f} min")
    _log(f"harvest done: {len(seen)} unique scenes -> {out}")


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def join(args) -> None:
    import numpy as np
    import pandas as pd
    from shapely import STRtree, points as shp_points
    from shapely import from_wkt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"reading {args.footprints}")
    fp = pd.read_csv(args.footprints)
    fp = fp[fp["wkt"].notna()].reset_index(drop=True)
    geoms = from_wkt(fp["wkt"].to_numpy())
    ok = np.array([g is not None and not g.is_empty for g in geoms])
    fp, geoms = fp[ok].reset_index(drop=True), geoms[ok]
    cloud = pd.to_numeric(fp["cloud"], errors="coerce").to_numpy()
    qual = pd.to_numeric(fp["quality"], errors="coerce").to_numpy()
    usable_scene = (cloud <= args.cloud_max) & ((qual == 0) | np.isnan(qual))
    _log(f"{len(fp)} scenes ({int(np.nansum(usable_scene))} usable: "
         f"cloud <= {args.cloud_max}%, nominal quality)")

    _log(f"reading {args.sites}")
    sites = pd.read_csv(args.sites)
    latc = args.lat_col if args.lat_col in sites.columns else _guess(sites, ("lat", "latitude", "y"))
    lonc = args.lon_col if args.lon_col in sites.columns else _guess(sites, ("lon", "longitude", "lng", "x"))
    idc = args.id_col if args.id_col in sites.columns else _guess(sites, ("site_id", "id", "Name", "name"))
    _log(f"columns: id={idc} lat={latc} lon={lonc}  ({len(sites)} sites)")
    lat = pd.to_numeric(sites[latc], errors="coerce").to_numpy()
    lon = pd.to_numeric(sites[lonc], errors="coerce").to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    if (~valid).any():
        _log(f"  dropping {int((~valid).sum())} sites without usable coordinates")
    sites = sites[valid].reset_index(drop=True)
    pts = shp_points(lon[valid], lat[valid])

    _log("building index and matching ...")
    tree = STRtree(geoms)
    pairs = tree.query(pts, predicate="within")  # (2, n) -> point idx, geom idx
    _log(f"{pairs.shape[1]} site-scene pairs")

    site_ids = sites[idc].to_numpy() if idc else np.arange(len(sites))
    pi, gi = pairs[0], pairs[1]
    pair_usable = usable_scene[gi]

    n_any = np.bincount(pi, minlength=len(sites))
    n_usable = np.bincount(pi[pair_usable], minlength=len(sites)) if pair_usable.any() else np.zeros(len(sites), int)

    # best (lowest-cloud) usable scene per site
    best_scene = np.full(len(sites), "", dtype=object)
    best_cloud = np.full(len(sites), np.nan)
    order = np.argsort(np.nan_to_num(cloud[gi], nan=999.0), kind="stable")
    for k in order[::-1]:  # reverse so the lowest cloud is written last
        if not pair_usable[k]:
            continue
        best_scene[pi[k]] = fp["scene_id"].iloc[gi[k]]
        best_cloud[pi[k]] = cloud[gi[k]]

    cov = pd.DataFrame({
        "site_id": site_ids,
        "lat": lat[valid], "lon": lon[valid],
        "n_scenes_any": n_any,
        "n_scenes_usable": n_usable,
        "covered_any": n_any > 0,
        "covered_usable": n_usable > 0,
        "best_scene_id": best_scene,
        "best_cloud": best_cloud,
    })
    cov_path = out_dir / "site_coverage.csv"
    cov.to_csv(cov_path, index=False)
    _log(f"wrote {cov_path}")

    # scene -> sites grouping, usable pairs only
    gdf = pd.DataFrame({"scene_id": fp["scene_id"].to_numpy()[gi[pair_usable]],
                        "site_id": site_ids[pi[pair_usable]]})
    grp = (gdf.groupby("scene_id")["site_id"]
              .agg(n_sites="count", site_ids=lambda s: ";".join(map(str, s)))
              .reset_index()
              .sort_values("n_sites", ascending=False))
    grp_path = out_dir / "scene_groups.csv"
    grp.to_csv(grp_path, index=False)
    _log(f"wrote {grp_path} ({len(grp)} scenes carrying >= 1 site)")

    stats = {
        "localities": args.localities,
        "sites": int(len(sites)),
        "scenes_harvested": int(len(fp)),
        "scenes_usable": int(np.nansum(usable_scene)),
        "covered_any": int((cov.covered_any).sum()),
        "covered_usable": int((cov.covered_usable).sum()),
        "scenes_with_sites": int(len(grp)),
        "cloud_max": args.cloud_max,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "funnel.json").write_text(json.dumps(stats, indent=2))
    print(funnel_text(stats, grp))
    (out_dir / "funnel.md").write_text(funnel_text(stats, grp))
    _log(f"wrote {out_dir / 'funnel.md'}")


def _guess(df, names) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def funnel_text(s: dict, grp=None) -> str:
    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "n/a"

    scenes = s["scenes_with_sites"]
    n = s["covered_usable"]
    lines = [
        "# Phase 0 funnel — EnMAP coverage of Mindat mine sites",
        "",
        f"Generated {s['generated']}. Usable = cloud <= {s['cloud_max']}% and nominal quality.",
        "",
        "| Stage | Count | Of previous |",
        "|---|---:|---:|",
        f"| Mindat localities | {s['localities']:,} | — |",
        f"| Unique sites (deduped coords) | {s['sites']:,} | {pct(s['sites'], s['localities'])} |",
        f"| Sites in >= 1 EnMAP scene | {s['covered_any']:,} | {pct(s['covered_any'], s['sites'])} |",
        f"| Sites with a usable scene | {s['covered_usable']:,} | {pct(s['covered_usable'], s['covered_any'])} |",
        f"| Distinct scenes to download | {scenes:,} | {(f'{n / scenes:.1f} sites/scene' if scenes else '—')} |",
        "",
        "## Storage, if every covered site becomes one 128x128 chip",
        "",
        "| Item | Size |",
        "|---|---:|",
        f"| Raw L2A archives ({scenes:,} scenes x {GB_RAW_PER_SCENE} GB) | {scenes * GB_RAW_PER_SCENE:,.0f} GB |",
        f"| Scene NetCDFs ({scenes:,} x {GB_NC_PER_SCENE} GB) | {scenes * GB_NC_PER_SCENE:,.0f} GB |",
        f"| Chips ({n:,} x {MB_CHIP} MB) | {n * MB_CHIP / 1024:,.0f} GB |",
        f"| **Final dataset** ({n:,} x ~{MB_SAMPLE} MB .npz) | **{n * MB_SAMPLE / 1024:,.0f} GB** |",
        "",
        f"Per-scene sizes measured on 2 scenes (128 SWIR bands, ~1180x1270 px). "
        f"The .npz figure is still an estimate. Raw archives and NetCDFs are intermediates "
        f"and can be deleted once the dataset is built.",
        "",
        f"At ~2 min to download plus ~2 min to process per scene, {scenes:,} scenes is "
        f"roughly {scenes * 4 / 60:,.0f} hours of single-threaded processing.",
    ]
    if grp is not None and len(grp):
        lines += ["", "## Densest scenes", "", "| Scene | Sites |", "|---|---:|"]
        for _, r in grp.head(10).iterrows():
            lines.append(f"| {r.scene_id} | {r.n_sites} |")
    return "\n".join(lines) + "\n"


def funnel_cmd(args) -> None:
    import pandas as pd
    out_dir = Path(args.out_dir)
    stats = json.loads((out_dir / "funnel.json").read_text())
    if args.localities:
        stats["localities"] = args.localities
    grp_path = out_dir / "scene_groups.csv"
    grp = pd.read_csv(grp_path) if grp_path.exists() else None
    txt = funnel_text(stats, grp)
    (out_dir / "funnel.md").write_text(txt)
    print(txt)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="page the whole EnMAP L2A collection")
    h.add_argument("--out", default="data/footprints.csv.gz")
    h.add_argument("--tile-deg", type=float, default=10.0)
    h.add_argument("--limit", type=int, default=PAGE_LIMIT, help="page size")
    h.add_argument("--start", default="2022-04-01")
    h.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    h.add_argument("--retries", type=int, default=3)
    h.add_argument("--restart", action="store_true", help="ignore saved tile progress")
    h.set_defaults(func=harvest)

    j = sub.add_parser("join", help="match sites to harvested footprints")
    j.add_argument("--sites", default="data/sites.csv")
    j.add_argument("--footprints", default="data/footprints.csv.gz")
    j.add_argument("--out-dir", default="data/coverage")
    j.add_argument("--cloud-max", type=float, default=10.0)
    j.add_argument("--localities", type=int, default=451221)
    j.add_argument("--id-col", default="site_id")
    j.add_argument("--lat-col", default="lat")
    j.add_argument("--lon-col", default="lon")
    j.set_defaults(func=join)

    f = sub.add_parser("funnel", help="re-print the report from existing outputs")
    f.add_argument("--out-dir", default="data/coverage")
    f.add_argument("--localities", type=int, default=None)
    f.set_defaults(func=funnel_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
