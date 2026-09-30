#!/usr/bin/env python3
"""
phase0_season.py
================

Re-picks which scene covers each site, preferring the local dry / low-vegetation
season. Nothing is downloaded: scene dates come from data/footprints.csv.gz and
the candidate (scene, site) pairs from scene_groups.csv.

Why: in a SWIR mineral chip you want rock, not chlorophyll. Late summer is
senesced-vegetation and snow-free almost everywhere outside the tropics, and a
roughly constant season across the dataset removes phenology as a source of
spectral variance between samples.

Hemisphere matters: "summer" is Jun-Sep north of the equator and Dec-Mar south
of it. Between +/-15 deg latitude season barely changes the spectrum, so those
sites are left unconstrained by default (--tropics-free).

Two modes:
  prefer (default)  cover in-season first, then fall back to any scene for the
                    sites that have no in-season option. Loses no sites.
  strict            in-season pairs only. Reports how many sites are lost.

Usage
-----
    python phase0_season.py                       # report + prefer ordering
    python phase0_season.py --mode strict
    python phase0_season.py --months-north 7,8,9 --months-south 1,2,3

Outputs (into --out-dir)
    season.md                    the trade-off: sites kept vs scenes needed
    scene_priority_season.csv    greedy scene order to feed build_chips_cog.py
    site_season.csv              per site: chosen scene, month, cloud, in_season
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase0_budget import greedy


def _log(msg: str) -> None:
    print(f"[season {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def _months(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--footprints", default="data/footprints.csv.gz")
    ap.add_argument("--scene-groups", default="data/coverage_v2_c50q2/scene_groups.csv")
    ap.add_argument("--sites", default="data/sites_v2.csv")
    ap.add_argument("--out-dir", default="data/coverage_v2_c50q2")
    ap.add_argument("--months-north", default="6,7,8,9")
    ap.add_argument("--months-south", default="12,1,2,3")
    ap.add_argument("--tropics-free", type=float, default=15.0,
                    help="within this |latitude|, any month counts as in-season")
    ap.add_argument("--mode", default="prefer", choices=["prefer", "strict"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    north, south = _months(args.months_north), _months(args.months_south)

    _log(f"reading {args.footprints}")
    fp = pd.read_csv(args.footprints, usecols=["scene_id", "datetime", "cloud"])
    dt = pd.to_datetime(fp["datetime"], errors="coerce", utc=True, format="mixed")
    fp["month"] = dt.dt.month
    fp["year"] = dt.dt.year
    month_of = dict(zip(fp.scene_id, fp.month))
    cloud_of = dict(zip(fp.scene_id, pd.to_numeric(fp.cloud, errors="coerce")))
    _log(f"{len(fp):,} scenes dated")

    sites = pd.read_csv(args.sites, low_memory=False)
    sites["site_id"] = sites.site_id.astype(str)
    lat_of = dict(zip(sites.site_id, pd.to_numeric(sites.lat, errors="coerce")))

    # candidate (scene, site) pairs -------------------------------------------
    grp = pd.read_csv(args.scene_groups)
    scene_sites: dict[str, set[str]] = {}
    site_scenes: dict[str, list[str]] = defaultdict(list)
    for sid, ids in zip(grp.scene_id, grp.site_ids.fillna("")):
        s = {x for x in str(ids).split(";") if x}
        if not s:
            continue
        scene_sites[sid] = s
        for x in s:
            site_scenes[x].append(sid)
    n_sites = len(site_scenes)
    n_pairs = sum(len(v) for v in scene_sites.values())
    _log(f"{n_sites:,} sites, {len(scene_sites):,} scenes, {n_pairs:,} usable pairs")

    def in_season(scene: str, site: str) -> bool:
        m = month_of.get(scene)
        lat = lat_of.get(site)
        if m is None or lat is None or np.isnan(lat):
            return False
        if abs(lat) <= args.tropics_free:
            return True
        return int(m) in (north if lat >= 0 else south)

    # per-site: is there an in-season scene, and which is the best one? -------
    chosen, rows = {}, []
    for site, scenes in site_scenes.items():
        ins = [s for s in scenes if in_season(s, site)]
        pool = ins or scenes
        best = min(pool, key=lambda s: (np.nan_to_num(cloud_of.get(s, np.nan), nan=999.0)))
        chosen[site] = best
        rows.append({"site_id": site, "lat": lat_of.get(site),
                     "n_scenes": len(scenes), "n_in_season": len(ins),
                     "chosen_scene": best, "chosen_month": month_of.get(best),
                     "chosen_cloud": cloud_of.get(best), "in_season": bool(ins)})
    per_site = pd.DataFrame(rows)
    per_site.to_csv(out_dir / "site_season.csv", index=False)
    have_season = set(per_site.loc[per_site.in_season, "site_id"])
    _log(f"{len(have_season):,} sites have at least one in-season scene "
         f"({100*len(have_season)/max(1,n_sites):.1f}%)")

    # greedy orderings --------------------------------------------------------
    season_pairs = {sid: {x for x in s if in_season(sid, x)}
                    for sid, s in scene_sites.items()}
    season_pairs = {k: v for k, v in season_pairs.items() if v}

    baseline = greedy(scene_sites, set(site_scenes))
    strict = greedy(season_pairs, have_season)

    if args.mode == "strict":
        order, kept = strict, have_season
    else:
        covered = {x for sid, _, _ in strict for x in season_pairs[sid]}
        rest = set(site_scenes) - covered
        tail = greedy(scene_sites, rest) if rest else []
        order = strict + tail
        kept = set(site_scenes)
        _log(f"prefer: {len(strict):,} in-season scenes cover {len(covered):,} sites; "
             f"{len(tail):,} more scenes for the remaining {len(rest):,}")

    pri = pd.DataFrame(order, columns=["scene_id", "new_sites", "cumulative_sites"])
    pri["rank"] = np.arange(1, len(pri) + 1)
    pri["month"] = [month_of.get(s) for s in pri.scene_id]
    pri["in_season_pick"] = [i < len(strict) for i in range(len(pri))]
    pri.to_csv(out_dir / "scene_priority_season.csv", index=False)

    # month histogram of what the chips would actually be cut from ------------
    picked_month = pd.Series([month_of.get(chosen[s]) for s in kept if s in chosen])
    hist = picked_month.value_counts().sort_index()

    md = [
        "# Phase 0 — season filter",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        f"In-season = month in {sorted(north)} north of the equator, "
        f"{sorted(south)} south of it; any month within "
        f"{args.tropics_free:.0f} deg of the equator.",
        "",
        "## Cost of the constraint",
        "",
        "| | Sites kept | Scenes to download |",
        "|---|---:|---:|",
        f"| No season constraint | {n_sites:,} | {len(baseline):,} |",
        f"| Season preferred (fallback allowed) | {len(site_scenes):,} | {len(order) if args.mode=='prefer' else '-'} |",
        f"| Season enforced (strict) | {len(have_season):,} | {len(strict):,} |",
        "",
        f"- Sites with no in-season scene at all: **{n_sites - len(have_season):,}** "
        f"({100*(n_sites-len(have_season))/max(1,n_sites):.1f}%). Strict mode drops these.",
        "",
        "## Month the chips would come from",
        "",
        "| Month | Sites |",
        "|---:|---:|",
    ]
    for m, n in hist.items():
        md.append(f"| {int(m) if pd.notna(m) else '?'} | {int(n):,} |")
    md += [
        "",
        "Snow and haze are recorded per chip by the builder (`qmask`), so any "
        "residual seasonal contamination is filterable after the fact rather "
        "than being a reason to rebuild.",
        "",
        "Feed the chosen ordering to the builder with:",
        "",
        "```",
        "python build_chips_cog.py --priority "
        f"{out_dir / 'scene_priority_season.csv'} --shard 0/4",
        "```",
    ]
    text = "\n".join(md) + "\n"
    (out_dir / "season.md").write_text(text)
    print(text)
    _log(f"wrote {out_dir/'season.md'}, scene_priority_season.csv, site_season.csv")


if __name__ == "__main__":
    main()
