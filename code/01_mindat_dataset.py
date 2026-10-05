#!/usr/bin/env python3
"""
01_mindat_dataset.py
====================

Step 1 of 3. Everything Mindat has to offer, cleaned into one point set with a
mineral list per point. No imagery is touched here.

    stage dump    pull the Mindat API into data/raw/ (localities, minerals,
                  locality types), resumable, skipped if the dump is already
                  there unless --force
    stage clean   raw -> data/sites.csv: drop points without coordinates,
                  drop the (0,0) placeholder, drop points with no mineral
                  list, merge duplicates that sit at the same position, and
                  annotate every point with its locality type and an is_mine
                  flag. No type filter: nothing is dropped for being a
                  "mountain" rather than a "mine".
    stage labels  count minerals over the sites, sweep the vocabulary
                  threshold, write data/labels/

Usage
-----
    python code/01_mindat_dataset.py --stage all
    python code/01_mindat_dataset.py --stage clean
    python code/01_mindat_dataset.py --stage labels --min-sites 100

Outputs
    data/raw/localities.jsonl.gz, geomaterials.csv, locality_types.csv
    data/sites.csv + data/sites.parquet
    data/labels/mineral_counts.csv, label_vocab.csv
    data/reports/01_clean.{json,md}, 01_labels.{json,md}, 01_merges.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TAG = "mindat"

API_BASE = "https://api.mindat.org"
ENDPOINTS = {                      # OpenMindat API, token auth
    "localities": "/localities/",
    "geomaterials": "/geomaterials/",
    "locality_types": "/locality-types/",
}

# A locality type counts as a mine if its name says so. Used as a FLAG, never
# as a filter: a point typed "mountain" can still carry a real mineral list.
MINE_WORDS = ("mine", "mining", "quarry", "open pit", "openpit", "pit", "shaft",
              "adit", "working", "prospect", "deposit", "tailing", "spoil",
              "dump", "placer", "borehole", "drill")

SWEEP = [25, 50, 100, 200, 500, 1000]

# Mindat writes an unknown position as exactly (0, 0) rather than leaving it
# empty, which is in the Gulf of Guinea and would otherwise look like a site.
NULL_ISLAND = 1e-6


# ==========================================================================
# stage: dump
# ==========================================================================
def paged(client, path: str, *, page_size: int, limit: int | None):
    """Yield every result from a paged OpenMindat endpoint."""
    url = f"{API_BASE}{path}"
    params = {"page_size": page_size}
    n = 0
    while url:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        rows = data.get("results", data if isinstance(data, list) else [])
        for row in rows:
            yield row
            n += 1
            if limit and n >= limit:
                return
        url = data.get("next") if isinstance(data, dict) else None
        params = None                        # `next` already carries them
        if not rows:
            return


def stage_dump(args) -> None:
    import httpx

    C.ensure_dirs()
    C.load_dotenv()
    token = C.env_any("MINDAT_API_KEY", "MINDAT_TOKEN", "OPENMINDAT_TOKEN")
    targets = {
        "localities": C.LOCALITIES_JSONL,
        "geomaterials": C.GEOMATERIALS_CSV,
        "locality_types": C.LOCALITY_TYPES_CSV,
    }
    missing = [k for k, p in targets.items() if not p.exists()]
    if not missing and not args.force:
        for k, p in targets.items():
            C.log(TAG, f"dump present, skipping {k}: {p.name}")
        return
    if not token:
        sys.exit("MINDAT_API_KEY not set in .env — needed to refresh the dump. "
                 "If you already have data/raw/localities.jsonl.gz, run "
                 "--stage clean instead.")

    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    with httpx.Client(timeout=120, headers=headers, follow_redirects=True) as client:
        # small reference tables first: they are what make the big one readable
        for key in ("geomaterials", "locality_types"):
            if targets[key].exists() and not args.force:
                continue
            C.log(TAG, f"fetching {key} ...")
            rows = list(paged(client, ENDPOINTS[key], page_size=1000, limit=None))
            pd.DataFrame(rows).to_csv(targets[key], index=False)
            C.log(TAG, f"  {len(rows):,} rows -> {targets[key].name}")

        if targets["localities"].exists() and not args.force:
            return
        C.log(TAG, "fetching localities (this is the long one) ...")
        tmp = targets["localities"].with_suffix(".part.gz")
        n = 0
        t0 = time.monotonic()
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for row in paged(client, ENDPOINTS["localities"],
                             page_size=args.page_size, limit=args.limit):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n % 10000 == 0:
                    rate = n / max(1e-6, time.monotonic() - t0)
                    C.log(TAG, f"  {n:,} localities ({rate:.0f}/s)")
        tmp.replace(targets["localities"])
        C.log(TAG, f"{n:,} localities -> {targets['localities'].name}")


# ==========================================================================
# stage: clean
# ==========================================================================
def _first(d: dict, names, default=None):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return default


def read_raw() -> pd.DataFrame:
    """Read the raw dump, keeping only the fields the dataset needs."""
    if not C.LOCALITIES_JSONL.exists():
        sys.exit(f"{C.LOCALITIES_JSONL} not found; run --stage dump first")
    rows = []
    with gzip.open(C.LOCALITIES_JSONL, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            gm = _first(d, ("geomaterials", "minerals", "geomaterial_ids"), [])
            if isinstance(gm, str):
                gm = C.split_list(gm)
            elif isinstance(gm, list):
                gm = [(g.get("id") if isinstance(g, dict) else g) for g in gm]
            rows.append({
                "site_id": _first(d, ("id", "loc_id", "locality_id")),
                "name": _first(d, ("txt", "name", "title"), ""),
                "lat": _first(d, ("latitude", "lat")),
                "lon": _first(d, ("longitude", "lon", "lng")),
                "locality_type": _first(d, ("locality_type", "lt_id", "type")),
                "description": _first(d, ("description_short", "description"), ""),
                "mineral_ids": [str(g) for g in (gm or []) if g not in (None, "")],
            })
    df = pd.DataFrame(rows)
    C.log(TAG, f"{len(df):,} raw localities")
    return df


def type_names() -> dict[str, str]:
    if not C.LOCALITY_TYPES_CSV.exists():
        C.log(TAG, f"no {C.LOCALITY_TYPES_CSV.name}; type names will be blank")
        return {}
    t = pd.read_csv(C.LOCALITY_TYPES_CSV)
    idc = next((c for c in ("id", "lt_id", "type_id") if c in t.columns), t.columns[0])
    nmc = next((c for c in ("txt", "name", "description", "title") if c in t.columns),
               t.columns[-1])
    return {str(k): str(v) for k, v in zip(t[idc], t[nmc])}


def mineral_names() -> dict[str, str]:
    if not C.GEOMATERIALS_CSV.exists():
        C.log(TAG, f"no {C.GEOMATERIALS_CSV.name}; mineral names will fall back to ids")
        return {}
    g = pd.read_csv(C.GEOMATERIALS_CSV, low_memory=False)
    idc = next((c for c in ("id", "gm_id", "mineral_id") if c in g.columns), g.columns[0])
    nmc = next((c for c in ("name", "txt", "title") if c in g.columns), g.columns[1])
    return {str(k): str(v) for k, v in zip(g[idc], g[nmc])}


def stage_clean(args) -> None:
    C.ensure_dirs()
    df = read_raw()
    funnel = [("raw localities", len(df))]

    # --- coordinates -------------------------------------------------------
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df[df.lat.notna() & df.lon.notna()]
    funnel.append(("has coordinates", len(df)))

    df = df[~((df.lat.abs() < NULL_ISLAND) & (df.lon.abs() < NULL_ISLAND))]
    funnel.append(("not the (0,0) placeholder", len(df)))

    df = df[(df.lat.abs() <= 90) & (df.lon.abs() <= 180)]
    funnel.append(("coordinates in range", len(df)))

    # --- the label has to exist -------------------------------------------
    df["n_minerals"] = df.mineral_ids.map(len)
    df = df[df.n_minerals > 0]
    funnel.append(("has a mineral list", len(df)))

    # --- annotate, do not filter ------------------------------------------
    tnames = type_names()
    df["locality_type"] = df.locality_type.map(lambda v: "" if pd.isna(v) else str(v))
    df["type_name"] = df.locality_type.map(lambda v: tnames.get(str(v), ""))
    low = df.type_name.str.lower()
    df["is_mine"] = low.apply(lambda s: any(w in s for w in MINE_WORDS))

    # --- merge points at the same position --------------------------------
    # A 64 px chip at 30 m spans ~1.9 km, so two localities 1 m apart are the
    # same pixel: keeping both would put identical imagery under two labels.
    df["coord_key"] = (df.lat.round(5).astype(str) + "," + df.lon.round(5).astype(str))
    before = len(df)
    df = df.sort_values("n_minerals", ascending=False)
    groups = df.groupby("coord_key", sort=False)

    merged_rows, merge_log = [], []
    for key, g in groups:
        head = g.iloc[0]
        ids: list[str] = []
        for lst in g.mineral_ids:
            for m in lst:
                if m not in ids:
                    ids.append(m)
        merged_rows.append({
            "site_id": head.site_id, "name": head["name"],
            "lat": head.lat, "lon": head.lon,
            "locality_type": head.locality_type, "type_name": head.type_name,
            "is_mine": bool(g.is_mine.any()),
            "mineral_ids": ids, "n_minerals": len(ids),
            "n_merged": len(g),
            "merged_site_ids": C.join_list(g.site_id.tolist()[1:]),
            "description": head.description, "coord_key": key,
        })
        if len(g) > 1:
            merge_log.append({
                "coord_key": key, "kept_site_id": head.site_id,
                "kept_name": head["name"], "n_merged": len(g),
                "merged_site_ids": C.join_list(g.site_id.tolist()[1:]),
                "merged_names": C.join_list(g["name"].tolist()[1:]),
                "minerals_after_merge": len(ids),
            })
    sites = pd.DataFrame(merged_rows)
    funnel.append(("after merging duplicate positions", len(sites)))
    C.log(TAG, f"merged {before - len(sites):,} duplicate-position rows "
               f"into {len(merge_log):,} groups")

    # --- names for the minerals -------------------------------------------
    gnames = mineral_names()
    sites["mineral_names"] = sites.mineral_ids.map(
        lambda ids: C.join_list(gnames.get(str(i), str(i)) for i in ids))
    sites["mineral_ids"] = sites.mineral_ids.map(C.join_list)
    sites = sites.sort_values("site_id").reset_index(drop=True)

    sites.to_csv(C.SITES_CSV, index=False)
    try:
        sites.to_parquet(C.DATA / "sites.parquet", index=False)
    except Exception as e:
        C.log(TAG, f"parquet skipped ({type(e).__name__})")
    pd.DataFrame(merge_log).to_csv(C.REPORTS / "01_merges.csv", index=False)
    C.log(TAG, f"wrote {C.SITES_CSV} ({len(sites):,} sites)")

    # --- report ------------------------------------------------------------
    types = (sites.groupby(["locality_type", "type_name"])
                  .size().sort_values(ascending=False).reset_index(name="sites"))
    kept = funnel[-1][1]
    stats = {
        "funnel": [{"stage": s, "rows": int(n)} for s, n in funnel],
        "sites": int(kept),
        "is_mine": int(sites.is_mine.sum()),
        "distinct_types": int(types.shape[0]),
        "merge_groups": int(len(merge_log)),
        "minerals_per_site": {
            "mean": float(sites.n_minerals.mean()),
            "median": float(sites.n_minerals.median()),
            "max": int(sites.n_minerals.max()),
        },
    }
    md = ["# Step 1 — Mindat point set", "",
          f"`{C.SITES_CSV.relative_to(C.REPO)}` holds **{kept:,} sites**, "
          f"{int(sites.is_mine.sum()):,} of them typed as mining-related. "
          f"No type filter was applied.", "", "## Funnel", ""]
    frows = []
    for i, (stage, n) in enumerate(funnel):
        lost = "" if i == 0 else f"-{funnel[i - 1][1] - n:,}"
        frows.append((stage, f"{n:,}", lost))
    md += C.md_table(["Stage", "Rows", "Lost"], frows, align="lrr")
    md += ["", "## Locality types present (top 20)", ""]
    md += C.md_table(["Type id", "Type", "Sites"],
                     [(r.locality_type, r.type_name or "(unnamed)", f"{int(r.sites):,}")
                      for r in types.head(20).itertuples()], align="llr")
    md += ["", f"Full type table in `reports/01_clean.json`. Merge groups: "
               f"`reports/01_merges.csv` ({len(merge_log):,} rows)."]
    stats["types"] = types.to_dict("records")
    C.write_report("01_clean", stats, "\n".join(md) + "\n")


# ==========================================================================
# stage: labels
# ==========================================================================
def stage_labels(args) -> None:
    C.ensure_dirs()
    if not C.SITES_CSV.exists():
        sys.exit(f"{C.SITES_CSV} not found; run --stage clean first")
    sites = pd.read_csv(C.SITES_CSV, low_memory=False)
    lists = [C.split_list(v) for v in sites.mineral_names]
    ids = [C.split_list(v) for v in sites.mineral_ids]

    counts: Counter = Counter()
    id_of: dict[str, str] = {}
    for names, mids in zip(lists, ids):
        uniq = set(names)
        counts.update(uniq)
        if len(mids) == len(names):
            for nm, mid in zip(names, mids):
                id_of.setdefault(nm, mid)

    df = pd.DataFrame({"mineral_name": list(counts)})
    df["mineral_id"] = df.mineral_name.map(id_of).fillna("")
    df["n_sites"] = df.mineral_name.map(counts)
    df = df.sort_values("n_sites", ascending=False).reset_index(drop=True)
    df["in_vocab"] = df.n_sites >= args.min_sites
    df.to_csv(C.LABELS / "mineral_counts.csv", index=False)
    vocab = df[df.in_vocab].copy()
    vocab.to_csv(C.LABELS / "label_vocab.csv", index=False)

    pool = [set(x) for x in lists]
    sweep = []
    for t in sorted(set(SWEEP + [args.min_sites])):
        v = set(df.loc[df.n_sites >= t, "mineral_name"])
        n = sum(1 for s in pool if s & v)
        sweep.append({"threshold": t, "minerals": len(v), "sites_with_label": n,
                      "pct_sites": 100 * n / max(1, len(pool))})

    C.log(TAG, f"{len(df):,} distinct minerals; {len(vocab):,} at >= {args.min_sites} sites")
    stats = {"sites": int(len(sites)), "distinct_minerals": int(len(df)),
             "min_sites": args.min_sites, "vocab_size": int(len(vocab)),
             "sweep": sweep,
             "labels_per_site": {
                 "mean": float(np.mean([len(s) for s in pool])),
                 "median": float(np.median([len(s) for s in pool])),
                 "max": int(max(len(s) for s in pool)) if pool else 0}}
    md = ["# Step 1 — label vocabulary", "",
          f"{len(sites):,} sites carry **{len(df):,} distinct minerals**. "
          f"The task is multi-label: a site's label is its whole mineral list, "
          f"and the list is presence-only — a mineral missing from it is not "
          f"evidence of absence.", "", "## Threshold sweep", ""]
    md += C.md_table(["Min sites", "Minerals in vocabulary", "Sites keeping >=1 label", "Share"],
                     [(r["threshold"], f"{r['minerals']:,}" + (" **<- chosen**" if r["threshold"] == args.min_sites else ""),
                       f"{r['sites_with_label']:,}", f"{r['pct_sites']:.1f}%") for r in sweep],
                     align="rrrr")
    md += ["", f"## Vocabulary at >= {args.min_sites} sites ({len(vocab):,} minerals)", ""]
    md += C.md_table(["Mineral", "Sites"],
                     [(r.mineral_name, f"{int(r.n_sites):,}") for r in vocab.head(40).itertuples()],
                     align="lr")
    if len(vocab) > 40:
        md.append(f"| ... {len(vocab) - 40:,} more | |")
    md += ["", "Counts here are over every site. Step 2 recounts them over the "
               "chips that actually got built, which is the vocabulary to train on."]
    C.write_report("01_labels", stats, "\n".join(md) + "\n")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "dump", "clean", "labels"])
    ap.add_argument("--force", action="store_true", help="re-download an existing dump")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="stop the dump after N rows")
    ap.add_argument("--min-sites", type=int, default=100,
                    help="a mineral enters the vocabulary at this many sites")
    args = ap.parse_args()

    if args.stage in ("all", "dump"):
        stage_dump(args)
    if args.stage in ("all", "clean"):
        stage_clean(args)
    if args.stage in ("all", "labels"):
        stage_labels(args)


if __name__ == "__main__":
    main()
