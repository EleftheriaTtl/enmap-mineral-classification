#!/usr/bin/env python3
"""
02_enmap_chips.py
=================

Step 2 of 3. For every site from step 1, find the EnMAP scene that covers it,
cut a 64x64 SWIR window around it, and save it with its mineral list. The
result is the labelled dataset.

    stage footprints  harvest every EnMAP L2A footprint from the DLR STAC
                      catalogue into data/interim/footprints.csv.gz. One pass
                      over 10-degree tiles, resumable; ~1 h cold.
    stage match       spatial join sites x footprints, then decide which one
                      scene each site is cut from. Season-aware and
                      scene-count-aware: a greedy cover picks as few scenes as
                      possible, then each site is assigned inside that set.
                      Writes data/interim/site_scene.csv.
    stage chips       read each chosen scene's Cloud-Optimized GeoTIFFs over
                      HTTP range requests and cut every assigned window.
                      Nothing but the chips touches disk. Resumable, shardable.
    stage index       walk the chips, write data/chips/index.csv and recount
                      the label vocabulary over what actually got built.

Why range reads rather than downloading archives: the COGs are 512x512
pixel-interleaved tiles, so one tile carries every band. The first chip in a
tile costs ~50 MB, every other chip in that tile is free while GDAL caches it.
Downloading the archives instead would be ~6.6 TB.

Usage
-----
    python code/02_enmap_chips.py --stage footprints
    python code/02_enmap_chips.py --stage match --cloud-max 50 --season prefer
    python code/02_enmap_chips.py --stage chips --limit 2 --verbose   # try it
    python code/02_enmap_chips.py --stage chips --shard 0/4           # for real
    python code/02_enmap_chips.py --stage index

Each chip is one .npz: X (int16, bands x 64 x 64), valid (uint8), qmask (the
five EnMAP quality layers, raw), wavelength, the mineral list, and the window's
CRS and affine transform so step 3 can rasterize onto the same grid.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TAG = "chips"


# ==========================================================================
# stage: footprints
# ==========================================================================
def tiles(deg: float) -> list[dict]:
    out, lat = [], -90.0
    while lat < 90.0:
        top = min(lat + deg, 90.0)
        lon = -180.0
        while lon < 180.0:
            right = min(lon + deg, 180.0)
            out.append({"name": f"{lat:+06.1f}_{lon:+07.1f}",
                        "bbox": [lon, lat, right, top]})
            lon = right
        lat = top
    return out


def page_tile(client, bbox, *, start: str, end: str, limit: int) -> Iterator[dict]:
    """Every item intersecting bbox, following whichever 'next' form the server uses."""
    body: dict[str, Any] = {"collections": [C.COLLECTION], "bbox": bbox, "limit": limit,
                            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"}
    url = f"{C.STAC_BASE}/search"
    seen: set[str] = set()
    while True:
        r = client.post(url, json=body, headers={"Content-Type": "application/json",
                                                 "Accept": "application/geo+json"})
        r.raise_for_status()
        data = r.json()
        feats = data.get("features") or []
        yield from feats
        nxt = next((l for l in (data.get("links") or []) if l.get("rel") == "next"), None)
        if not nxt or not feats:
            return
        if nxt.get("body"):
            merged = {**body, **nxt["body"]} if nxt.get("merge", True) else dict(nxt["body"])
            token = json.dumps(merged, sort_keys=True)
            if token in seen:
                return
            seen.add(token)
            body, url = merged, nxt.get("href") or url
        elif nxt.get("href"):
            href = nxt["href"]
            while href and href not in seen:
                seen.add(href)
                rr = client.get(href, headers={"Accept": "application/geo+json"})
                rr.raise_for_status()
                data = rr.json()
                yield from (data.get("features") or [])
                nx = next((l for l in (data.get("links") or []) if l.get("rel") == "next"), None)
                href = nx.get("href") if nx else None
            return
        else:
            return


def scene_props(f: dict) -> tuple[float | None, int | None, str]:
    p = f.get("properties") or {}
    cc = p.get("eo:cloud_cover", p.get("cloudCover", p.get("cloud_cover")))
    q = p.get("enmap:overallQuality", p.get("enmap:OverallQuality"))
    try:
        cc = float(cc) if cc is not None else None
    except (TypeError, ValueError):
        cc = None
    try:
        q = int(q) if q is not None else None
    except (TypeError, ValueError):
        q = None
    return cc, q, str(p.get("datetime") or "")


def geom_wkt(f: dict) -> str | None:
    from shapely.geometry import box, shape

    g = f.get("geometry")
    if not g:
        b = f.get("bbox")
        return box(b[0], b[1], b[2], b[3]).wkt if b and len(b) >= 4 else None
    try:
        return shape(g).wkt
    except Exception:
        return None


def stage_footprints(args) -> None:
    import httpx

    C.ensure_dirs()
    out = C.FOOTPRINTS_CSV
    state = out.with_suffix(out.suffix + ".tiles.json")
    done: set[str] = set()
    if state.exists() and not args.restart:
        done = set(json.loads(state.read_text()).get("done", []))
        C.log(TAG, f"resuming: {len(done)} tiles already harvested")

    seen: set[str] = set()
    mode = "a" if (out.exists() and done and not args.restart) else "w"
    if mode == "a":
        with gzip.open(out, "rt", encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                seen.add(line.split(",", 1)[0])
        C.log(TAG, f"resuming: {len(seen):,} scenes already on disk")

    todo = [c for c in tiles(args.tile_deg) if c["name"] not in done]
    C.log(TAG, f"{len(todo)} tiles to go ({args.start}..{args.end})")

    t0, consec = time.monotonic(), 0
    with gzip.open(out, mode + "t", encoding="utf-8") as fh, httpx.Client(timeout=120) as client:
        if mode == "w":
            fh.write("scene_id,datetime,cloud,quality,wkt\n")
        for i, cell in enumerate(todo, 1):
            ok = False
            for attempt in range(1, args.retries + 1):
                try:
                    for f in page_tile(client, cell["bbox"], start=args.start,
                                       end=args.end, limit=args.page_limit):
                        sid = f.get("id")
                        if not sid or sid in seen:
                            continue
                        wkt = geom_wkt(f)
                        if not wkt:
                            continue
                        cc, q, dt = scene_props(f)
                        seen.add(sid)
                        fh.write(f'{sid},{dt},{"" if cc is None else cc},'
                                 f'{"" if q is None else q},"{wkt}"\n')
                    ok = True
                    break
                except Exception as e:
                    msg = str(e)
                    if "recovery mode" in msg or "NoApplicableCode" in msg:
                        C.log(TAG, "the STAC catalogue reports a server-side database "
                                   "failure. Stopping; progress is saved, rerun later.")
                        return
                    if attempt == args.retries:
                        C.log(TAG, f"  tile {cell['name']}: giving up ({e!r})")
                    else:
                        time.sleep(args.backoff * attempt)
            consec = 0 if ok else consec + 1
            if consec >= args.max_consecutive_failures:
                C.log(TAG, f"{consec} tiles failed in a row; stopping. Rerun later.")
                return
            done.add(cell["name"])
            fh.flush()
            state.write_text(json.dumps({"done": sorted(done)}))
            if args.sleep:
                time.sleep(args.sleep)
            if i % 10 == 0 or i == len(todo):
                rate = i / max(1e-6, time.monotonic() - t0)
                C.log(TAG, f"  {i}/{len(todo)} tiles, {len(seen):,} scenes, "
                           f"ETA {(len(todo) - i) / max(1e-6, rate) / 60:.0f} min")
    C.log(TAG, f"harvest done: {len(seen):,} unique scenes -> {out.name}")


# ==========================================================================
# stage: match
# ==========================================================================
def greedy_cover(scene_sites: dict[str, set[str]], wanted: set[str]) -> list[str]:
    """Scenes most-new-sites-first, until `wanted` is covered."""
    import heapq

    heap = [(-len(s & wanted), sid) for sid, s in scene_sites.items()]
    heap = [h for h in heap if -h[0] > 0]
    heapq.heapify(heap)
    covered: set[str] = set()
    order: list[str] = []
    while heap:
        neg, sid = heapq.heappop(heap)
        gain = len(scene_sites[sid] & (wanted - covered))
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:          # stale: reinsert with the true gain
            heapq.heappush(heap, (-gain, sid))
            continue
        covered |= scene_sites[sid] & wanted
        order.append(sid)
    return order


def stage_match(args) -> None:
    from shapely import STRtree, from_wkt, points as shp_points

    C.ensure_dirs()
    if not C.FOOTPRINTS_CSV.exists():
        sys.exit(f"{C.FOOTPRINTS_CSV} not found; run --stage footprints first")
    if not C.SITES_CSV.exists():
        sys.exit(f"{C.SITES_CSV} not found; run step 01 first")

    C.log(TAG, f"reading {C.FOOTPRINTS_CSV.name}")
    fp = pd.read_csv(C.FOOTPRINTS_CSV)
    fp = fp[fp.wkt.notna()].reset_index(drop=True)
    geoms = from_wkt(fp.wkt.to_numpy())
    ok = np.array([g is not None and not g.is_empty for g in geoms])
    fp, geoms = fp[ok].reset_index(drop=True), geoms[ok]
    cloud = pd.to_numeric(fp.cloud, errors="coerce").to_numpy()
    qual = pd.to_numeric(fp.quality, errors="coerce").to_numpy()
    month = pd.to_datetime(fp.datetime, errors="coerce", utc=True,
                           format="mixed").dt.month.to_numpy()
    usable = (cloud <= args.cloud_max) & ((qual <= args.max_quality) | np.isnan(qual))
    C.log(TAG, f"{len(fp):,} scenes, {int(np.nansum(usable)):,} usable "
               f"(cloud <= {args.cloud_max}%)")

    sites = pd.read_csv(C.SITES_CSV, low_memory=False)
    sites["site_id"] = sites.site_id.astype(str)
    lat = pd.to_numeric(sites.lat, errors="coerce").to_numpy()
    lon = pd.to_numeric(sites.lon, errors="coerce").to_numpy()
    good = np.isfinite(lat) & np.isfinite(lon)
    sites, lat, lon = sites[good].reset_index(drop=True), lat[good], lon[good]
    C.log(TAG, f"{len(sites):,} sites")

    C.log(TAG, "indexing footprints and matching ...")
    tree = STRtree(geoms)
    pi, gi = tree.query(shp_points(lon, lat), predicate="within")
    keep = usable[gi]
    pi, gi = pi[keep], gi[keep]
    C.log(TAG, f"{len(pi):,} usable site-scene pairs")

    site_ids = sites.site_id.to_numpy()
    scene_ids = fp.scene_id.to_numpy()

    # in-season, hemisphere aware. Within the tropics the season barely changes
    # the spectrum, so those sites are left unconstrained.
    north = {int(x) for x in args.months_north.split(",")}
    south = {int(x) for x in args.months_south.split(",")}
    plat = lat[pi]
    pmonth = month[gi]
    in_season = np.where(
        np.abs(plat) <= args.tropics_free, True,
        np.where(plat >= 0,
                 np.isin(pmonth, list(north)),
                 np.isin(pmonth, list(south))))
    in_season = np.where(np.isnan(pmonth.astype(float)), False, in_season)

    scene_all: dict[str, set[str]] = defaultdict(set)
    scene_ssn: dict[str, set[str]] = defaultdict(set)
    cand: dict[str, list[int]] = defaultdict(list)
    for k in range(len(pi)):
        s, sc = site_ids[pi[k]], scene_ids[gi[k]]
        scene_all[sc].add(s)
        if in_season[k]:
            scene_ssn[sc].add(s)
        cand[s].append(k)
    covered = set(cand)
    have_season = {s for s, ks in cand.items() if any(in_season[k] for k in ks)}
    C.log(TAG, f"{len(covered):,} sites covered; {len(have_season):,} have an "
               f"in-season scene ({100 * len(have_season) / max(1, len(covered)):.1f}%)")

    # Pick as few scenes as possible, in-season first. Assigning every site its
    # own lowest-cloud scene independently would multiply the scene count and so
    # the download time, for no extra chips.
    if args.season == "strict":
        chosen = greedy_cover(scene_ssn, have_season)
        target = have_season
    else:
        chosen = greedy_cover(scene_ssn, have_season)
        rest = covered - {s for sc in chosen for s in scene_ssn[sc]}
        if rest:
            chosen += greedy_cover(scene_all, rest)
        target = covered
    chosen_set = set(chosen)
    C.log(TAG, f"{len(chosen):,} scenes chosen for {len(target):,} sites")

    # assign each site inside the chosen set: in-season first, then least cloud
    rows = []
    for s, ks in cand.items():
        if s not in target:
            continue
        opts = [k for k in ks if scene_ids[gi[k]] in chosen_set]
        if not opts:
            continue
        best = min(opts, key=lambda k: (not bool(in_season[k]),
                                        np.nan_to_num(cloud[gi[k]], nan=999.0)))
        rows.append({"site_id": s, "scene_id": scene_ids[gi[best]],
                     "scene_month": None if np.isnan(float(pmonth[best])) else int(pmonth[best]),
                     "scene_cloud": float(cloud[gi[best]]) if not np.isnan(cloud[gi[best]]) else None,
                     "in_season": bool(in_season[best]),
                     "n_candidate_scenes": len(ks)})
    assign = pd.DataFrame(rows)
    meta = sites.set_index("site_id")
    for col in ("lat", "lon", "mineral_names", "mineral_ids", "n_minerals",
                "type_name", "is_mine", "name"):
        if col in meta.columns:
            assign[col] = assign.site_id.map(meta[col])
    assign.to_csv(C.SITE_SCENE_CSV, index=False)
    C.log(TAG, f"wrote {C.SITE_SCENE_CSV.name} ({len(assign):,} sites)")

    pri = (assign.groupby("scene_id").size().sort_values(ascending=False)
                 .reset_index(name="n_sites"))
    pri["rank"] = np.arange(1, len(pri) + 1)
    pri["cumulative_sites"] = pri.n_sites.cumsum()
    pri.to_csv(C.SCENE_PRIORITY_CSV, index=False)

    mhist = assign.scene_month.value_counts().sort_index()
    stats = {
        "sites_in": int(len(sites)), "scenes_harvested": int(len(fp)),
        "scenes_usable": int(np.nansum(usable)), "cloud_max": args.cloud_max,
        "max_quality": args.max_quality, "pairs": int(len(pi)),
        "sites_covered": int(len(covered)), "sites_with_in_season": int(len(have_season)),
        "season_mode": args.season, "months_north": sorted(north),
        "months_south": sorted(south), "tropics_free_deg": args.tropics_free,
        "sites_assigned": int(len(assign)), "scenes_chosen": int(len(pri)),
        "in_season_share": float(assign.in_season.mean()) if len(assign) else 0.0,
        "month_histogram": {int(k): int(v) for k, v in mhist.items()},
        "chips_per_scene": {"mean": float(pri.n_sites.mean()) if len(pri) else 0,
                            "max": int(pri.n_sites.max()) if len(pri) else 0},
    }
    md = ["# Step 2 — scene matching", "",
          f"**{len(assign):,} sites** will be cut from **{len(pri):,} scenes**. "
          f"{100 * assign.in_season.mean():.0f}% of them from an in-season scene.",
          "", "## Funnel", ""]
    md += C.md_table(["Stage", "Count"], [
        ("Sites from step 1", f"{len(sites):,}"),
        ("EnMAP scenes in the catalogue", f"{len(fp):,}"),
        (f"Usable (cloud <= {args.cloud_max}%)", f"{int(np.nansum(usable)):,}"),
        ("Sites with >=1 usable scene", f"{len(covered):,}"),
        ("Sites with an in-season option", f"{len(have_season):,}"),
        ("Sites assigned a scene", f"{len(assign):,}"),
        ("Scenes to read", f"{len(pri):,}"),
    ], align="lr")
    md += ["", "## Month the chips come from", ""]
    md += C.md_table(["Month", "Sites"],
                     [(int(k), f"{int(v):,}") for k, v in mhist.items()], align="rr")
    md += ["", f"Season rule: months {sorted(north)} north of the equator, "
               f"{sorted(south)} south, unconstrained within "
               f"{args.tropics_free:.0f} deg of it. Mode `{args.season}`.",
           "", "Scene count is kept low by a greedy cover; each site is then "
               "assigned one scene inside that set, so no site is chipped twice."]
    C.write_report("02_match", stats, "\n".join(md) + "\n")


# ==========================================================================
# stage: chips
# ==========================================================================
def process_scene(scene_id: str, want: pd.DataFrame, *, chip_px: int, version: str,
                  max_patch_cloud: float, mask_layers: tuple[str, ...],
                  min_valid: float, verbose: bool) -> tuple[int, int, int, str]:
    """Cut every assigned window out of one scene. Returns (written, edge, dropped, error)."""
    import rasterio
    from rasterio.windows import Window

    urls = C.scene_urls(scene_id)
    scene_dir = C.CHIPS / scene_id
    written = edge = dropped = 0
    bands, wl = C.band_plan(urls["metadata"], version)

    with rasterio.open("/vsicurl/" + C.with_auth(urls["image"])) as src:
        quality = {}
        for a in C.QUALITY_ASSETS:
            try:
                quality[a] = rasterio.open("/vsicurl/" + C.with_auth(urls[a]))
            except Exception:
                pass                                   # a missing layer is not fatal
        try:
            half = chip_px // 2
            nd = src.nodatavals[0] if src.nodatavals else None
            for _, s in want.iterrows():
                out = scene_dir / f"{s.site_id}.npz"
                if out.exists():
                    continue
                row, col = src.index(*C.to_scene_xy(src.crs, float(s.lon), float(s.lat)))
                r0, c0 = int(row) - half, int(col) - half
                if r0 < 0 or c0 < 0 or r0 + chip_px > src.height or c0 + chip_px > src.width:
                    edge += 1
                    continue
                win = Window(c0, r0, chip_px, chip_px)

                # read every quality layer raw; decide what counts as bad after
                qvals = {a: q.read(1, window=win).astype(np.uint8)
                         for a, q in quality.items()}
                cl = qvals.get("QL_QUALITY_CLOUD")
                cloud_frac = float((cl != 0).mean()) if cl is not None else 0.0
                if cloud_frac > max_patch_cloud:
                    dropped += 1
                    if verbose:
                        C.log(TAG, f"      {s.site_id}: patch cloud {cloud_frac:.0%}, dropped")
                    continue

                X = src.read(bands.tolist(), window=win)
                bad = ((X == nd).all(axis=0) if nd is not None
                       else np.zeros((chip_px, chip_px), dtype=bool))
                for name in mask_layers:
                    a = C.SHORT[name]
                    if a in qvals:
                        bad |= qvals[a] != 0
                valid = (~bad).astype(np.uint8)
                vfrac = float(valid.mean())
                if vfrac < min_valid:
                    dropped += 1
                    if verbose:
                        C.log(TAG, f"      {s.site_id}: valid {vfrac:.0%} < "
                                   f"{min_valid:.0%}, dropped")
                    continue

                names = [C.LONG[a] for a in C.QUALITY_ASSETS if a in qvals]
                qmask = (np.stack([qvals[C.SHORT[n]] for n in names]) if names
                         else np.zeros((0, chip_px, chip_px), dtype=np.uint8))
                pfrac = np.array([float((qvals[C.SHORT[n]] != 0).mean()) for n in names],
                                 dtype=np.float32)
                wtr = src.window_transform(win)

                scene_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out,
                    # X is exactly what the sensor gave: `valid` says what to trust,
                    # so a masking decision can be revisited without rebuilding.
                    X=X.astype(np.int16), valid=valid, qmask=qmask,
                    qmask_names=np.array(names), patch_frac=pfrac,
                    mask_layers=np.array(list(mask_layers)),
                    wavelength=wl.astype(np.float32),
                    site_id=str(s.site_id), lat=float(s.lat), lon=float(s.lon),
                    minerals=str(s.get("mineral_names", "")),
                    mineral_ids=str(s.get("mineral_ids", "")),
                    is_mine=bool(s.get("is_mine", False)),
                    scene_id=scene_id, patch_cloud=np.float32(cloud_frac),
                    # grid, so step 3 can rasterize OSM onto these same pixels
                    crs_wkt=str(src.crs.to_wkt()),
                    transform=np.array(list(wtr)[:6], dtype=np.float64),
                    chip_px=np.int16(chip_px), pixel_m=np.float32(C.PIXEL_M),
                    scale_factor=np.float32(1e-4), nodata=np.int16(C.NODATA),
                )
                written += 1
                if verbose:
                    flags = " ".join(f"{n[:5]} {f:.0%}" for n, f in zip(names, pfrac) if f > 0)
                    C.log(TAG, f"      {s.site_id}: {X.shape} cloud {cloud_frac:.0%} "
                               f"valid {vfrac:.0%}" + (f"  [{flags}]" if flags else ""))
        finally:
            for q in quality.values():
                q.close()
    return written, edge, dropped, ""


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*.npz")) / 1e9


def stage_chips(args) -> None:
    C.ensure_dirs()
    C.dlr_auth()
    C.gdal_env(args.cache_mb)
    if not C.SITE_SCENE_CSV.exists():
        sys.exit(f"{C.SITE_SCENE_CSV} not found; run --stage match first")

    raw = [x.strip().lower() for x in args.mask_layers.split(",") if x.strip()]
    mask_layers = () if raw == ["none"] else tuple(raw)
    unknown = [x for x in mask_layers if x not in C.SHORT]
    if unknown:
        sys.exit(f"--mask-layers: unknown {unknown}; pick from {sorted(C.SHORT)} or 'none'")
    C.log(TAG, f"masking on {list(mask_layers) or 'nothing'}; all five layers are stored")

    assign = pd.read_csv(C.SITE_SCENE_CSV, low_memory=False)
    assign["site_id"] = assign.site_id.astype(str)
    by_scene = {sid: g for sid, g in assign.groupby("scene_id")}
    order = pd.read_csv(C.SCENE_PRIORITY_CSV).scene_id.tolist() \
        if C.SCENE_PRIORITY_CSV.exists() else list(by_scene)
    scenes = [s for s in order if s in by_scene]

    tag = ""
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        scenes = scenes[i::n]
        tag = f"_{i}of{n}"
        C.log(TAG, f"shard {i}/{n}: {len(scenes):,} scenes")
    if args.limit:
        scenes = scenes[: args.limit]

    manifest = C.CHIPS / f"manifest{tag}.csv"
    done: set[str] = set()
    if manifest.exists():
        m = pd.read_csv(manifest)
        done = set(m.loc[m.status == "ok", "scene_id"].astype(str))
        C.log(TAG, f"resuming: {len(done):,} scenes already built")

    rows, consec, total, t0 = [], 0, 0, time.monotonic()
    for k, sid in enumerate(scenes, 1):
        if sid in done:
            continue
        gb = dir_size_gb(C.CHIPS)
        if gb >= args.max_gb:
            C.log(TAG, f"dataset reached {gb:.1f} GB (--max-gb); stopping")
            break
        want = by_scene[sid]
        C.log(TAG, f"{k}/{len(scenes)} {sid[:44]}  {len(want)} chips  [{gb:.1f} GB]")
        t = time.monotonic()
        try:
            w, e, d, err = process_scene(
                sid, want, chip_px=args.chip_px, version=args.version,
                max_patch_cloud=args.max_patch_cloud, mask_layers=mask_layers,
                min_valid=args.min_valid, verbose=args.verbose)
            total += w
            rows.append({"scene_id": sid, "status": "ok" if not err else "error",
                         "chips": w, "edge": e, "dropped": d, "error": err,
                         "seconds": round(time.monotonic() - t, 1)})
            C.log(TAG, f"      {w} written, {e} off-edge, {d} dropped, "
                       f"{time.monotonic() - t:.0f}s")
            consec = 0
        except KeyboardInterrupt:
            break
        except Exception as e:
            name = type(e).__name__
            rows.append({"scene_id": sid, "status": "error", "chips": 0, "edge": 0,
                         "dropped": 0, "error": repr(e)[:300], "seconds": 0})
            if any(x in name for x in ("Timeout", "Connect", "Network", "SSL")):
                consec += 1
                C.log(TAG, f"      network error ({name}); {consec} in a row")
                if consec >= args.max_net_failures:
                    C.log(TAG, "the download host is unreachable; progress is saved, rerun later.")
                    break
            else:
                C.log(TAG, f"      ERROR {e!r}")
                traceback.print_exc()
        if len(rows) >= 5:
            pd.DataFrame(rows).to_csv(manifest, mode="a",
                                      header=not manifest.exists(), index=False)
            rows = []
    if rows:
        pd.DataFrame(rows).to_csv(manifest, mode="a",
                                  header=not manifest.exists(), index=False)
    mins = (time.monotonic() - t0) / 60
    C.log(TAG, f"done: {total:,} chips this run, {dir_size_gb(C.CHIPS):.1f} GB total, "
               f"{mins:.0f} min ({total / max(1e-6, mins):.0f} chips/min)")


# ==========================================================================
# stage: index
# ==========================================================================
def stage_index(args) -> None:
    C.ensure_dirs()
    files = sorted(C.CHIPS.glob("*/*.npz"))
    if not files:
        sys.exit(f"no chips under {C.CHIPS}; run --stage chips first")
    C.log(TAG, f"indexing {len(files):,} chips")

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=False)          # npz is lazy: X is not read
        rows.append({
            "site_id": str(d["site_id"]), "path": str(f.relative_to(C.DATA)),
            "scene_id": str(d["scene_id"]),
            "lat": float(d["lat"]), "lon": float(d["lon"]),
            "minerals": str(d["minerals"]),
            "n_minerals": len(C.split_list(str(d["minerals"]))),
            "is_mine": bool(d["is_mine"]) if "is_mine" in d else None,
            "patch_cloud": float(d["patch_cloud"]),
            "valid_frac": float(d["valid"].mean()),
            "bands": int(d["wavelength"].shape[0]),
            "chip_px": int(d["chip_px"]) if "chip_px" in d else None,
            "bytes": f.stat().st_size,
        })
    idx = pd.DataFrame(rows)
    idx["month"] = [(C.scene_date(s).month if C.scene_date(s) else None)
                    for s in idx.scene_id]
    idx = idx.sort_values("site_id").reset_index(drop=True)
    idx.to_csv(C.CHIP_INDEX_CSV, index=False)
    C.log(TAG, f"wrote {C.CHIP_INDEX_CSV.name}")

    # recount the vocabulary over what actually exists
    from collections import Counter

    counts: Counter = Counter()
    for v in idx.minerals:
        counts.update(set(C.split_list(v)))
    built = pd.DataFrame({"mineral_name": list(counts),
                          "n_chips": [counts[m] for m in counts]})
    built = built.sort_values("n_chips", ascending=False).reset_index(drop=True)
    built["in_vocab"] = built.n_chips >= args.min_sites
    built.to_csv(C.LABELS / "mineral_counts_built.csv", index=False)

    dup = int(idx.site_id.duplicated().sum())
    stats = {
        "chips": int(len(idx)), "sites": int(idx.site_id.nunique()),
        "duplicate_site_ids": dup,
        "scenes": int(idx.scene_id.nunique()),
        "gb": round(float(idx.bytes.sum()) / 1e9, 2),
        "mb_per_chip": round(float(idx.bytes.mean()) / 1e6, 3),
        "bands": int(idx.bands.mode().iloc[0]),
        "chip_px": int(idx.chip_px.mode().iloc[0]) if idx.chip_px.notna().any() else None,
        "valid_frac": {"mean": float(idx.valid_frac.mean()),
                       "p05": float(idx.valid_frac.quantile(0.05)),
                       "fully_masked": int((idx.valid_frac == 0).sum())},
        "patch_cloud": {"mean": float(idx.patch_cloud.mean()),
                        "zero": int((idx.patch_cloud == 0).sum())},
        "distinct_minerals": int(len(built)),
        "vocab_size": int(built.in_vocab.sum()), "min_sites": args.min_sites,
        "month_histogram": {int(k): int(v) for k, v in
                            idx.month.value_counts().sort_index().items()},
    }
    md = ["# Step 2 — the labelled dataset", "",
          f"**{len(idx):,} chips** from {idx.scene_id.nunique():,} scenes, "
          f"{stats['gb']:.1f} GB ({stats['mb_per_chip']:.2f} MB each), "
          f"{stats['bands']} bands at {stats['chip_px']}x{stats['chip_px']} px.", ""]
    if dup:
        md += [f"> {dup:,} site ids appear more than once — a site was cut from "
               f"several scenes. Expected only if the assignment was bypassed.", ""]
    md += ["## Quality", ""]
    md += C.md_table(["Measure", "Value"], [
        ("Mean valid fraction", f"{stats['valid_frac']['mean']:.1%}"),
        ("5th percentile valid", f"{stats['valid_frac']['p05']:.1%}"),
        ("Fully masked chips", f"{stats['valid_frac']['fully_masked']:,}"),
        ("Mean patch cloud", f"{stats['patch_cloud']['mean']:.1%}"),
        ("Cloud-free patches", f"{stats['patch_cloud']['zero']:,}"),
    ], align="lr")
    md += ["", f"## Labels over the built chips ({len(built):,} minerals, "
               f"{int(built.in_vocab.sum()):,} at >= {args.min_sites} chips)", ""]
    md += C.md_table(["Mineral", "Chips"],
                     [(r.mineral_name, f"{int(r.n_chips):,}")
                      for r in built.head(30).itertuples()], align="lr")
    md += ["", f"Index: `{C.CHIP_INDEX_CSV.relative_to(C.REPO)}` — one row per chip, "
               f"with its path, label, cloud and valid fraction. This is the file to "
               f"load for training."]
    C.write_report("02_dataset", stats, "\n".join(md) + "\n")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "footprints", "match", "chips", "index"])
    # footprints
    ap.add_argument("--tile-deg", type=float, default=10.0)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--page-limit", type=int, default=500)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--restart", action="store_true")
    # match
    ap.add_argument("--cloud-max", type=float, default=50.0)
    ap.add_argument("--max-quality", type=int, default=2, choices=[0, 1, 2],
                    help="EnMAP overallQuality; 2 = any. It duplicates the cloud "
                         "filter, so restricting it mostly costs sites.")
    ap.add_argument("--season", default="prefer", choices=["prefer", "strict", "off"])
    ap.add_argument("--months-north", default="6,7,8,9")
    ap.add_argument("--months-south", default="12,1,2,3")
    ap.add_argument("--tropics-free", type=float, default=15.0)
    # chips
    ap.add_argument("--chip-px", type=int, default=64)
    ap.add_argument("--version", default="128", choices=["v1", "128", "58"])
    ap.add_argument("--max-patch-cloud", type=float, default=0.2)
    ap.add_argument("--mask-layers", default="cloud,cloudshadow",
                    help="folded into `valid`: cloud,cloudshadow,haze,cirrus,snow "
                         "or 'none'. All five are stored raw either way.")
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--max-gb", type=float, default=130.0)
    ap.add_argument("--cache-mb", type=int, default=2048)
    ap.add_argument("--max-net-failures", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    # index
    ap.add_argument("--min-sites", type=int, default=100)
    args = ap.parse_args()

    if args.season == "off":
        args.months_north = ",".join(str(m) for m in range(1, 13))
        args.months_south = args.months_north
        args.season = "prefer"

    if args.stage in ("all", "footprints"):
        stage_footprints(args)
    if args.stage in ("all", "match"):
        stage_match(args)
    if args.stage in ("all", "chips"):
        stage_chips(args)
    if args.stage in ("all", "index"):
        stage_index(args)


if __name__ == "__main__":
    main()
