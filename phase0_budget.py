#!/usr/bin/env python3
"""
phase0_budget.py
================

Turns the coverage tables into a scoping decision:

1. Spatial thinning. sites.csv is deduped at ~1 m, but a 128x128 chip at 30 m
   covers 3.8 km. Sites closer together than that share a chip, so they are
   near-duplicate samples. Thinning keeps one site per --thin-m grid cell
   (preferring the site with the most minerals listed).

2. Scene budget. Scene density is very skewed, so a greedy pick (most new
   sites first) shows how many sites you get for N scenes downloaded. That is
   the curve that answers "what do we get for 1 TB?".

Usage
-----
    python phase0_budget.py --coverage data/coverage/site_coverage.csv \
        --scene-groups data/coverage/scene_groups.csv \
        --sites data/sites.csv --vocab data/labels/label_vocab.csv \
        --out-dir data/coverage

Outputs
    budget.md              the curve + recommended operating points
    scene_priority.csv     scenes in greedy order, with cumulative sites
    sites_thinned.csv      the thinned site list (one per grid cell)
"""
from __future__ import annotations

import argparse
import heapq
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

GB_RAW = 0.45
GB_NC = 0.21
MB_CHIP = 10.5
MB_SAMPLE = 6.0
MIN_PER_SCENE = 4.0  # ~2 min download + ~2 min processing
CHECKPOINTS = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000]


def _log(msg: str) -> None:
    print(f"[budget {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def thin(df: pd.DataFrame, metres: float, rank: np.ndarray) -> pd.DataFrame:
    """Keep one site per ~metres grid cell; highest rank wins."""
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    dlat = metres / 111_320.0
    # cell width in longitude varies with latitude; use the cell's own latitude
    lat_cell = np.floor(lat / dlat).astype(np.int64)
    coslat = np.maximum(np.cos(np.radians(lat)), 0.01)
    dlon = dlat / coslat
    lon_cell = np.floor(lon / dlon).astype(np.int64)
    key = pd.Series(list(zip(lat_cell, lon_cell)))
    order = np.argsort(-rank, kind="stable")
    keep = (pd.DataFrame({"key": key.to_numpy()[order]})
            .assign(pos=order)
            .drop_duplicates("key")["pos"].to_numpy())
    return df.iloc[np.sort(keep)].reset_index(drop=True)


def greedy(scene_sites: dict[str, set[str]], wanted: set[str]) -> list[tuple[str, int, int]]:
    """Pick scenes most-new-sites-first. Returns (scene_id, new, cumulative)."""
    heap = [(-len(s & wanted), sid) for sid, s in scene_sites.items()]
    heap = [h for h in heap if -h[0] > 0]
    heapq.heapify(heap)
    covered: set[str] = set()
    out: list[tuple[str, int, int]] = []
    while heap:
        neg, sid = heapq.heappop(heap)
        gain = len(scene_sites[sid] & wanted - covered)
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:       # stale entry, re-insert with the true gain
            heapq.heappush(heap, (-gain, sid))
            continue
        covered |= scene_sites[sid] & wanted
        out.append((sid, gain, len(covered)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", default="data/coverage/site_coverage.csv")
    ap.add_argument("--scene-groups", default="data/coverage/scene_groups.csv")
    ap.add_argument("--sites", default="data/sites.csv")
    ap.add_argument("--vocab", default=None, help="label_vocab.csv: restrict to sites with a label")
    ap.add_argument("--out-dir", default="data/coverage")
    ap.add_argument("--thin-m", type=float, default=1280.0,
                    help="grid size for thinning; 1280 m = a third of a 3.8 km chip")
    ap.add_argument("--name-col", default="mineral_names")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"reading {args.coverage}")
    cov = pd.read_csv(args.coverage)
    cov = cov[cov["covered_usable"].astype(str).str.lower().isin(["true", "1"])].reset_index(drop=True)
    n_covered = len(cov)
    _log(f"{n_covered:,} sites with a usable scene")

    sites = pd.read_csv(args.sites, low_memory=False)
    id_col = "site_id" if "site_id" in sites.columns else sites.columns[0]
    n_min = pd.to_numeric(sites.get("n_minerals"), errors="coerce").fillna(0).to_numpy()
    if args.name_col in sites.columns:
        minerals = dict(zip(sites[id_col], sites[args.name_col].fillna("")))
    else:
        alt = next((c for c in ("mineral_names", "geomaterials", "minerals") if c in sites.columns), None)
        if alt:
            _log(f"no column {args.name_col!r}; using {alt!r} for mineral names")
        minerals = dict(zip(sites[id_col], sites[alt].fillna(""))) if alt else {}
    rank_by_id = dict(zip(sites[id_col], n_min))

    labelled: set[str] | None = None
    if args.vocab:
        vocab = set(pd.read_csv(args.vocab)["mineral_name"])
        def has_label(sid):
            names = str(minerals.get(sid, ""))
            return any(m in names for m in vocab) if names else False
        labelled = {sid for sid in cov["site_id"] if has_label(sid)}
        _log(f"{len(labelled):,} covered sites carry at least one vocabulary label")

    rank = np.array([rank_by_id.get(s, 0) for s in cov["site_id"]], dtype=float)
    if labelled is not None:
        rank = rank + np.array([1000.0 if s in labelled else 0.0 for s in cov["site_id"]])

    thinned = thin(cov, args.thin_m, rank)
    thinned.to_csv(out_dir / "sites_thinned.csv", index=False)
    _log(f"thinning at {args.thin_m:.0f} m: {n_covered:,} -> {len(thinned):,} sites "
         f"({100 * len(thinned) / n_covered:.1f}%)")

    grp = pd.read_csv(args.scene_groups)
    keep_ids = set(thinned["site_id"].astype(str))
    if labelled is not None:
        keep_ids &= {str(s) for s in labelled}
        _log(f"after also requiring a label: {len(keep_ids):,} sites")
    scene_sites = {}
    for sid, ids in zip(grp["scene_id"], grp["site_ids"].fillna("")):
        s = {x for x in str(ids).split(";") if x in keep_ids}
        if s:
            scene_sites[sid] = s
    _log(f"{len(scene_sites):,} scenes carry at least one kept site")

    order = greedy(scene_sites, keep_ids)
    pri = pd.DataFrame(order, columns=["scene_id", "new_sites", "cumulative_sites"])
    pri["rank"] = np.arange(1, len(pri) + 1)
    pri.to_csv(out_dir / "scene_priority.csv", index=False)

    total = pri["cumulative_sites"].iloc[-1] if len(pri) else 0
    rows = []
    for n in [c for c in CHECKPOINTS if c < len(pri)] + [len(pri)]:
        got = int(pri["cumulative_sites"].iloc[n - 1])
        rows.append({
            "scenes": n, "sites": got, "pct": 100 * got / max(1, total),
            "raw_tb": n * GB_RAW / 1024, "nc_tb": n * GB_NC / 1024,
            "final_gb": got * MB_SAMPLE / 1024,
            "hours": n * MIN_PER_SCENE / 60,
            "days_8x": n * MIN_PER_SCENE / 60 / 8 / 24,
        })
    tbl = pd.DataFrame(rows)

    md = [
        "# Phase 0 — scene budget and thinning",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Thinning",
        "",
        f"- Covered sites (usable scene): **{n_covered:,}**",
        f"- After thinning to one site per {args.thin_m:.0f} m cell: **{len(thinned):,}** "
        f"({100 * len(thinned) / n_covered:.1f}%)",
    ]
    if labelled is not None:
        md.append(f"- Also requiring a vocabulary label: **{len(keep_ids):,}**")
    md += [
        "",
        f"A 128x128 chip at 30 m spans 3.8 km, so sites closer than that share almost the same "
        f"pixels. Thinning removes near-duplicate samples and prevents the same ground appearing "
        f"in both training and test splits.",
        "",
        "## What N scenes buys you",
        "",
        "Scenes picked most-new-sites-first.",
        "",
        "| Scenes | Sites | Share | Raw download | NetCDF | Final dataset | 1 core | 8 parallel |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in tbl.iterrows():
        md.append(f"| {int(r.scenes):,} | {int(r.sites):,} | {r.pct:.0f}% | {r.raw_tb:.2f} TB | "
                  f"{r.nc_tb:.2f} TB | {r.final_gb:.0f} GB | {r.hours:,.0f} h | {r.days_8x:.1f} d |")
    md += [
        "",
        "Raw archives and NetCDFs are transient: process one scene, keep the chips, delete the "
        "archive. Peak disk is then roughly one scene plus the chips, not the totals above.",
        "",
        "## Reading the curve",
        "",
    ]
    if len(tbl) > 2:
        half = tbl[tbl.pct >= 50]
        knee = half.iloc[0] if len(half) else tbl.iloc[-1]
        md.append(f"- Half the sites come from the first **{int(knee.scenes):,} scenes** "
                  f"({knee.raw_tb:.2f} TB of downloads, {knee.hours:,.0f} core-hours).")
        ninety = tbl[tbl.pct >= 90]
        if len(ninety):
            k9 = ninety.iloc[0]
            md.append(f"- 90% needs **{int(k9.scenes):,} scenes** ({k9.raw_tb:.2f} TB).")
        md.append(f"- The last 10% is the long tail: mostly scenes carrying a single site.")
    md += [
        "",
        f"Per-scene sizes measured on 2 real scenes; the {MB_SAMPLE} MB per sample is still an "
        f"estimate, so the final-dataset column has roughly +/-30% on it.",
    ]
    text = "\n".join(md) + "\n"
    (out_dir / "budget.md").write_text(text)
    print(text)
    _log(f"wrote {out_dir / 'budget.md'} and scene_priority.csv")


if __name__ == "__main__":
    main()
