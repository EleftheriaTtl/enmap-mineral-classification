#!/usr/bin/env python3
"""
01_stac_search_download.py
==========================

Step 1 of the EnMAP mineral-classification pipeline.

For every (mine, lat, lon) row in `data/<mineral>/localities.csv`, query the
DLR EOWEB GeoPortal STAC catalog for an EnMAP L2A scene that:

    * geographically intersects (lat, lon)
    * has  cloudCover           <  10%      (overridable)
    * has  nominalQuality flag <= 0         (i.e. nominal quality)
    * lies inside [--start, --end]          (default: full archive)

The single best match per mine (lowest cloud %, then most recent) is written
to `data/<mineral>/stac_matches.csv` and the L2A asset is downloaded to
`data/<mineral>/raw/`.

Authentication
--------------
EOWEB downloads require a registered DLR account. Set in `.env`:

    EOWEB_USER=...
    EOWEB_PASS=...
    EOWEB_BASE_URL=https://geoservice.dlr.de/eoc/ogc/stac/v1   # default
    EOWEB_DOWNLOAD_AUTH=basic                                  # basic|bearer

Usage
-----
    python 01_stac_search_download.py --mineral magnesite
    python 01_stac_search_download.py --mineral galena \
        --start 2023-01-01 --end 2024-12-31 --cloud-max 5

Notes
-----
* The script is conservative: it never overwrites an already-downloaded zip.
* If a row already has a successful download recorded in stac_matches.csv,
  it is skipped (resume-by-default).
* If no scene matches a row, the row is recorded with status = "no_match"
  so step 02+ can ignore it cleanly.
"""
from __future__ import annotations

# IMPORTANT: print something immediately so the user knows the script is alive
# even before the heavy imports (pandas/requests) finish loading. Without this,
# the terminal can look frozen for 1-3 seconds at startup.
import sys as _sys, time as _time, faulthandler as _fh
_sys.stderr.write(f"[stac {_time.strftime('%H:%M:%S')}] booting...\n")
_sys.stderr.flush()
# Ctrl+\ (SIGQUIT on macOS/Linux) will now dump the full Python stack to stderr
# so you can see exactly where it's stuck.
_fh.enable()

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import pipeline_utils as pu
import geoserver_download as gsd

sys.stderr.write(f"[stac {_time.strftime('%H:%M:%S')}] imports done\n")
sys.stderr.flush()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[stac {_ts()}] {msg}", file=sys.stderr, flush=True)

DEFAULT_STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
DEFAULT_COLLECTION = "ENMAP_HSI_L2A"
DEFAULT_START = "2022-04-01"  # EnMAP launch
USER_AGENT = "enmap-mineral-classification/1.0"


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------
def stac_search_point(
    *,
    base_url: str,
    collection: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    cloud_max: float,
    limit: int = 100,
    session: requests.Session,
) -> list[dict]:
    """
    POST to <base_url>/search and return matching items as a list of dicts.

    We let the server filter by bbox + datetime; the cloud / quality filters
    are applied client-side because not every STAC implementation supports
    CQL2 properly.
    """
    # GeoServer STAC endpoint may return 401 if you send wrong BasicAuth headers
    # even when anonymous search is allowed. We therefore use the vendored
    # GeoServer downloader's anonymous STAC search implementation.
    geom = {"type": "Point", "coordinates": [lon, lat]}
    feats = gsd.search_stac(
        geom,
        collection=collection,
        datetime_range=f"{start}T00:00:00Z/{end}T23:59:59Z",
        limit=limit,
        max_cloud_cover=cloud_max,
        # Keep legacy behavior: prefer nominal scenes. If property missing, we accept.
        # The EnMAP collection exposes enmap:overallQuality (0 nominal, 1 reduced, 2 low).
        max_overall_quality=0,
        sortby_cloud_asc=True,
        logger=None,
    )

    # Defensive + STRICT filtering client-side (server-side query support can vary).
    keep: list[dict] = []
    for f in feats:
        if not (isinstance(f, dict) and f.get("type") == "Feature"):
            continue
        props = f.get("properties", {}) or {}

        # 1) strict cloud filter
        cc = props.get("eo:cloud_cover", props.get("cloudCover", props.get("cloud_cover")))
        if cc is None:
            # if cloud cover missing, skip (strict)
            continue
        try:
            if float(cc) > float(cloud_max):
                continue
        except (TypeError, ValueError):
            continue

        # 2) strict nominal quality filter (0 only)
        oq = props.get("enmap:overallQuality", props.get("enmap:OverallQuality", None))
        if oq is not None:
            try:
                if int(oq) > 0:
                    continue
            except (TypeError, ValueError):
                continue

        keep.append(f)

    return keep


def best_scene(features: list[dict]) -> Optional[dict]:
    """Lowest cloud %, then most recent datetime."""
    if not features:
        return None

    def key(f: dict):
        props = f.get("properties", {}) or {}
        cc = props.get("eo:cloud_cover", props.get("cloudCover", props.get("cloud_cover")))
        try:
            cc_v = float(cc) if cc is not None else 100.0
        except (TypeError, ValueError):
            cc_v = 100.0
        dt = props.get("datetime") or "1970-01-01T00:00:00Z"
        try:
            ts = datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = 0.0
        return (cc_v, -ts)

    return sorted(features, key=key)[0]


# ---------------------------------------------------------------------------
# Asset selection + download
# ---------------------------------------------------------------------------
ASSET_KEY_PREFERENCE = ("data", "L2A", "enmap_l2a", "product", "metadata")


def pick_l2a_asset(item: dict) -> Optional[tuple[str, str]]:
    """
    Return (asset_key, href) for the L2A archive (zip / tar.gz) of an item.
    Preference order: known keys -> any zip -> any tar.gz.
    """
    assets: dict = item.get("assets", {}) or {}

    # 1) preferred keys
    for k in ASSET_KEY_PREFERENCE:
        if k in assets:
            href = assets[k].get("href")
            if href:
                return k, href

    # 2) anything ending .zip / .ZIP / .tar.gz
    for k, a in assets.items():
        href = (a or {}).get("href", "")
        if href.lower().endswith((".zip", ".tar.gz", ".tgz")):
            return k, href

    return None


def download(
    *, url: str, dest: Path, session: requests.Session, auth: Optional[tuple[str, str]]
) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        _log(f"      already on disk: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.monotonic()
    with session.get(url, auth=auth, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        if total:
            _log(f"      downloading {dest.name}  ({total/1e6:.1f} MB)")
        else:
            _log(f"      downloading {dest.name}  (size unknown)")
        got = 0
        next_pct = 10
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = int(got * 100 / total)
                    if pct >= next_pct:
                        rate = got / max(1e-6, time.monotonic() - t0) / 1e6
                        _log(f"      ... {pct}%  ({got/1e6:.1f}/{total/1e6:.1f} MB, {rate:.1f} MB/s)")
                        next_pct += 10
    tmp.rename(dest)
    _log(f"      saved {dest.name} in {time.monotonic()-t0:.1f}s")
    return dest


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    *,
    mineral: str,
    start: str,
    end: str,
    cloud_max: float,
    base_url: str,
    collection: str,
    skip_download: bool,
    limit: Optional[int] = None,
    target: Optional[int] = None,
) -> None:
    _log("loading .env ...")
    pu.load_dotenv()
    # Credentials are only required for downloads (SSO), not for STAC search.
    # If missing and downloads are enabled, geoserver_download will fail with an
    # actionable message.
    user = os.environ.get("ENMAP_USERNAME") or os.environ.get("EOWEB_USER")
    pwd = os.environ.get("ENMAP_PASSWORD") or os.environ.get("EOWEB_PASS")
    auth_mode = os.environ.get("EOWEB_DOWNLOAD_AUTH", "basic").lower()

    _log("creating http session ...")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    # IMPORTANT: Do NOT set session.auth here. If credentials are wrong, some
    # deployments will return 401 even though anonymous STAC search is allowed.

    download_auth: Optional[tuple[str, str]] = (user, pwd) if (auth_mode == "basic" and user and pwd) else None

    _log(f"loading localities for {mineral} ...")
    df = pu.load_localities(mineral)
    out_csv = pu.stac_matches_csv(mineral)
    raw = pu.raw_dir(mineral)

    _log(f"config: base_url={base_url} collection={collection}")
    _log(f"config: window={start}..{end}  cloud_max={cloud_max}%  user={user}")
    _log(f"input:  {len(df)} localities from {pu.localities_csv(mineral)}")
    _log(f"output: {out_csv}  raw_dir={raw}")

    # Probe the STAC endpoint ONCE so connectivity / auth issues fail fast and
    # loudly instead of looking like an infinite hang on the first locality.
    try:
        _log(f"probing STAC search (anonymous)  POST {base_url.rstrip('/')}/search  (10s timeout)")
        t0 = time.monotonic()
        # tiny probe: global point, 1 result
        feats = gsd.search_stac(
            {"type": "Point", "coordinates": [0.0, 0.0]},
            collection=collection,
            datetime_range=f"{start}T00:00:00Z/{end}T23:59:59Z",
            limit=1,
            logger=None,
        )
        _log(f"probe -> OK in {time.monotonic()-t0:.2f}s (returned {len(feats)} feature(s))")
    except requests.exceptions.ConnectTimeout as e:
        _log(f"probe TIMED OUT connecting to {base_url}: {e}")
        _log("network or DNS issue. Try `curl -v <base_url>` to confirm.")
        raise SystemExit(2)
    except requests.exceptions.ConnectionError as e:
        _log(f"probe CONNECTION ERROR: {e!r}")
        raise SystemExit(2)
    except Exception as e:
        _log(f"probe FAILED: {e!r}")
        raise

    # resume support
    prev: dict[str, dict] = {}
    if out_csv.exists():
        for r in pd.read_csv(out_csv).to_dict("records"):
            prev[str(r.get("Name", ""))] = r
        _log(f"resume: {len(prev)} previous rows in stac_matches.csv")

    if limit is not None and limit > 0:
        df = df.head(limit).copy()
        _log(f"--limit {limit}  -> processing first {len(df)} localities only")

    matches: list[dict] = []
    n_ok = 0  # localities with a downloaded scene (= future samples)
    if target:
        _log(f"--target {target}  -> stop once {target} localities have a downloaded scene")
    pipeline_t0 = time.monotonic()
    n = len(df)
    for i, row in df.reset_index(drop=True).iterrows():
        name = str(row["Name"])
        lat, lon = float(row["lat"]), float(row["lon"])
        row_t0 = time.monotonic()

        if target and n_ok >= target:
            _log(f"target reached ({n_ok}/{target} localities with a scene) -> stopping")
            break

        # Heartbeat BEFORE any network call so the user always knows where we are.
        _log(f"{i+1}/{n} START  {name!r}  @ lat={lat:.5f},lon={lon:.5f}")

        if name in prev and str(prev[name].get("status", "")) == "downloaded":
            _log(f"{i+1}/{n} SKIP   already downloaded ({prev[name].get('scene_id')})")
            matches.append(prev[name])
            n_ok += 1
            continue

        try:
            search_t0 = time.monotonic()
            feats = stac_search_point(
                base_url=base_url,
                collection=collection,
                lat=lat,
                lon=lon,
                start=start,
                end=end,
                cloud_max=cloud_max,
                session=session,
            )
            _log(
                f"      search  -> {len(feats)} candidate(s) "
                f"in {time.monotonic()-search_t0:.1f}s"
            )
        except requests.HTTPError as e:
            _log(f"{i+1}/{n} HTTP   {e} for {name}")
            matches.append({"Name": name, "lat": lat, "lon": lon, "status": "search_error"})
            continue
        except Exception as e:
            _log(f"{i+1}/{n} ERROR  search failed: {e!r}")
            matches.append({"Name": name, "lat": lat, "lon": lon, "status": "search_error"})
            continue

        item = best_scene(feats)
        if item is None:
            matches.append({"Name": name, "lat": lat, "lon": lon, "status": "no_match"})
            _log(f"{i+1}/{n} NONE   no scene matched filters  ({time.monotonic()-row_t0:.1f}s)")
            continue

        props = item.get("properties", {}) or {}
        overall_q = props.get("enmap:overallQuality", props.get("enmap:OverallQuality", None))
        rec = {
            "Name": name,
            "lat": lat,
            "lon": lon,
            "scene_id": item.get("id"),
            "datetime": props.get("datetime"),
            "cloud_cover": props.get(
                "eo:cloud_cover", props.get("cloudCover", props.get("cloud_cover"))
            ),
            # Keep legacy column name, but prefer EnMAP overall quality when present:
            # 0=nominal, 1=reduced, 2=low. (We enforce <=0 upstream.)
            "nominal_quality": (overall_q if overall_q is not None else props.get("nominalQuality", props.get("quality:nominal", 0))),
            "status": "matched",
            "asset_key": None,
            "asset_url": None,
            "raw_path": None,
        }
        _log(
            f"      best    {item.get('id')}  cloud={rec['cloud_cover']}%  "
            f"datetime={rec['datetime']}"
        )
        picked = pick_l2a_asset(item)
        if picked is None:
            rec["status"] = "no_asset"
            matches.append(rec)
            _log(f"{i+1}/{n} NOASSET no L2A asset on {item.get('id')}")
            continue
        asset_key, href = picked
        rec["asset_key"] = asset_key
        rec["asset_url"] = href

        if skip_download:
            rec["status"] = "search_only"
            matches.append(rec)
            _log(f"{i+1}/{n} DRY    --skip-download set, not fetching")
            continue

        # download
        fname = href.rstrip("/").split("/")[-1].split("?")[0]
        if not fname.lower().endswith((".zip", ".tar.gz", ".tgz")):
            fname = f"{item.get('id')}.zip"
        dest = raw / fname
        try:
            # Prefer the GeoServer download helper: it handles SSO redirects and
            # writes a stable <item_id>.zip with EnMAP-like filenames inside.
            existing = raw / f"{item.get('id')}.zip"
            if existing.exists() and existing.stat().st_size > 0:
                # scene already fetched for another locality -> reuse it
                _log(f"      reuse   {existing.name} (already downloaded for another locality)")
                ok = True
            else:
                ok = gsd.download_item(item, raw, username=user, password=pwd, logger=None)
            if ok:
                n_ok += 1
                rec["status"] = "downloaded"
                rec["raw_path"] = str((raw / f"{item.get('id')}.zip").relative_to(pu.REPO_ROOT))
            else:
                rec["status"] = "download_error: auth_or_empty_zip"
            _log(
                f"{i+1}/{n} OK     {dest.name}  cloud={rec['cloud_cover']}%  "
                f"row_elapsed={time.monotonic()-row_t0:.1f}s"
            )
        except Exception as e:
            rec["status"] = f"download_error: {e!r}"
            _log(f"{i+1}/{n} FAIL   download failed: {e!r}")

        matches.append(rec)

        # checkpoint every 10 rows
        if (i + 1) % 10 == 0:
            _write(out_csv, matches)
            done = i + 1
            elapsed = time.monotonic() - pipeline_t0
            rate = done / max(1e-6, elapsed)
            eta = (n - done) / max(1e-6, rate)
            _log(
                f"checkpoint: {done}/{n} rows  ({rate:.2f} rows/s, ETA ~{eta/60:.1f} min)  "
                f"-> {out_csv.name}"
            )

    _write(out_csv, matches)
    n_dl = sum(1 for r in matches if r.get("status") == "downloaded")
    n_no = sum(1 for r in matches if r.get("status") == "no_match")
    n_err = sum(1 for r in matches if "error" in str(r.get("status", "")))
    _log(
        f"DONE: {n_dl} downloaded, {n_no} no_match, {n_err} errors  "
        f"(total elapsed {time.monotonic()-pipeline_t0:.1f}s) -> {out_csv}"
    )


def _write(out_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "Name",
        "lat",
        "lon",
        "scene_id",
        "datetime",
        "cloud_cover",
        "nominal_quality",
        "status",
        "asset_key",
        "asset_url",
        "raw_path",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser(description="Search EOWEB STAC and download EnMAP L2A scenes.")
    ap.add_argument("--mineral", required=True, help="Mineral name (folder under data/).")
    ap.add_argument("--start", default=DEFAULT_START, help=f"YYYY-MM-DD (default {DEFAULT_START}).")
    ap.add_argument("--end", default=today, help="YYYY-MM-DD (default today).")
    ap.add_argument("--cloud-max", type=float, default=10.0, help="Max cloud cover %% (default 10).")
    ap.add_argument(
        "--base-url",
        default=__import__("os").environ.get("EOWEB_BASE_URL", DEFAULT_STAC_URL),
        help="STAC base URL.",
    )
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="STAC collection id.")
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Only resolve matches; do not actually download (useful for dry runs).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N localities (debug / sanity-check).",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=None,
        help="Stop once N localities have a downloaded scene (e.g. 100 samples).",
    )
    args = ap.parse_args()

    run(
        mineral=args.mineral,
        start=args.start,
        end=args.end,
        cloud_max=args.cloud_max,
        base_url=args.base_url,
        collection=args.collection,
        skip_download=args.skip_download,
        limit=args.limit,
        target=args.target,
    )


if __name__ == "__main__":
    main()
