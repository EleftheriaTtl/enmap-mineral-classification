#!/usr/bin/env python3
"""
phase0_labels.py
================

Phase 0, step 2: count sites per mineral and choose the label vocabulary.

Counts each mineral over all sites and, when a coverage table exists, over the
covered sites only — the vocabulary should be set on minerals you can actually
get imagery for. Prints a threshold sweep so the cut-off is a decision, not a
guess, then writes the vocabulary at --min-sites.

Usage
-----
    # all sites only (before the coverage step)
    python phase0_labels.py --sites data/sites.csv

    # with coverage (preferred): counts on sites that have a usable scene
    python phase0_labels.py --sites data/sites.csv \
        --coverage data/coverage/site_coverage.csv --min-sites 100

Outputs (in --out-dir)
    mineral_counts.csv   one row per mineral: n_sites, n_covered_sites, in_vocab
    label_vocab.csv      the minerals at or above the threshold
    label_report.md      threshold sweep + head of the vocabulary
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

SWEEP = [25, 50, 100, 200, 500, 1000]


def _log(msg: str) -> None:
    print(f"[labels {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def split_cell(v, sep: Optional[str]) -> List[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s or s in {"[]", "nan"}:
        return []
    if s.startswith("[") and s.endswith("]"):  # a python-ish list repr
        parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
        if parts:
            return [a or b for a, b in parts]
        s = s[1:-1]
    if sep:
        parts = s.split(sep)
    else:
        for cand in (";", "|", ","):
            if cand in s:
                parts = s.split(cand)
                break
        else:
            parts = [s]
    return [p.strip().strip("'\"") for p in parts if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="data/sites.csv")
    ap.add_argument("--coverage", default=None, help="site_coverage.csv from phase0_coverage.py")
    ap.add_argument("--out-dir", default="data/labels")
    ap.add_argument("--min-sites", type=int, default=100, help="vocabulary threshold")
    ap.add_argument("--name-col", default="mineral_names")
    ap.add_argument("--id-col", default="mineral_ids")
    ap.add_argument("--site-id-col", default="site_id")
    ap.add_argument("--sep", default=None, help="separator inside the list columns (auto by default)")
    ap.add_argument("--coverage-col", default="covered_usable",
                    choices=["covered_usable", "covered_any"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"reading {args.sites}")
    sites = pd.read_csv(args.sites)
    if args.name_col not in sites.columns:
        sys.exit(f"no column {args.name_col!r} in {args.sites}; columns are {list(sites.columns)}")
    _log(f"{len(sites):,} sites")

    covered_mask = None
    if args.coverage:
        cov = pd.read_csv(args.coverage)
        if args.site_id_col in sites.columns and "site_id" in cov.columns:
            good = set(cov.loc[cov[args.coverage_col].astype(str).str.lower().isin(["true", "1"]), "site_id"])
            covered_mask = sites[args.site_id_col].isin(good).to_numpy()
        else:
            if len(cov) != len(sites):
                sys.exit("cannot align coverage with sites: no shared id column and different lengths")
            covered_mask = cov[args.coverage_col].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        _log(f"{int(covered_mask.sum()):,} sites are covered ({args.coverage_col})")

    names = [split_cell(v, args.sep) for v in sites[args.name_col]]
    ids = [split_cell(v, args.sep) for v in sites[args.id_col]] if args.id_col in sites.columns else None

    all_c: Counter = Counter()
    cov_c: Counter = Counter()
    id_of: dict[str, str] = {}
    for i, row in enumerate(names):
        uniq = set(row)
        all_c.update(uniq)
        if covered_mask is not None and covered_mask[i]:
            cov_c.update(uniq)
        if ids is not None and len(ids[i]) == len(row):
            for nm, mid in zip(row, ids[i]):
                id_of.setdefault(nm, mid)

    df = pd.DataFrame({"mineral_name": list(all_c)})
    df["mineral_id"] = df["mineral_name"].map(id_of).fillna("")
    df["n_sites"] = df["mineral_name"].map(all_c)
    if covered_mask is not None:
        df["n_covered_sites"] = df["mineral_name"].map(lambda m: cov_c.get(m, 0))
    key = "n_covered_sites" if covered_mask is not None else "n_sites"
    df = df.sort_values(key, ascending=False).reset_index(drop=True)
    df["in_vocab"] = df[key] >= args.min_sites
    df.to_csv(out_dir / "mineral_counts.csv", index=False)
    _log(f"wrote {out_dir / 'mineral_counts.csv'} ({len(df):,} minerals)")

    # threshold sweep: how many labels, and how many sites keep at least one
    pool = [set(r) for i, r in enumerate(names) if covered_mask is None or covered_mask[i]]
    rows = []
    for t in sorted(set(SWEEP + [args.min_sites])):
        vocab = set(df.loc[df[key] >= t, "mineral_name"])
        labelled = sum(1 for s in pool if s & vocab)
        rows.append({"threshold": t, "minerals": len(vocab),
                     "sites_with_label": labelled,
                     "pct_sites": 100 * labelled / max(1, len(pool))})
    sweep = pd.DataFrame(rows)

    vocab_df = df[df["in_vocab"]].copy()
    vocab_df.to_csv(out_dir / "label_vocab.csv", index=False)

    scope = "covered sites" if covered_mask is not None else "all sites"
    md = [
        "# Phase 0 — label vocabulary",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Counts over **{scope}** "
        f"({len(pool):,} sites, {len(df):,} distinct minerals).",
        "",
        "## Threshold sweep",
        "",
        "| Min sites | Minerals in vocabulary | Sites keeping >= 1 label | Share |",
        "|---:|---:|---:|---:|",
    ]
    for _, r in sweep.iterrows():
        mark = "  **<- chosen**" if int(r.threshold) == args.min_sites else ""
        md.append(f"| {int(r.threshold)} | {int(r.minerals)}{mark} | {int(r.sites_with_label):,} | {r.pct_sites:.1f}% |")
    md += [
        "",
        f"## Vocabulary at >= {args.min_sites} sites ({len(vocab_df)} minerals)",
        "",
        "| Mineral | Sites |",
        "|---|---:|",
    ]
    for _, r in vocab_df.head(40).iterrows():
        md.append(f"| {r.mineral_name} | {int(r[key]):,} |")
    if len(vocab_df) > 40:
        md.append(f"| ... {len(vocab_df) - 40} more | |")
    md += ["", f"Full list: `{out_dir / 'label_vocab.csv'}`; all minerals: `{out_dir / 'mineral_counts.csv'}`."]
    text = "\n".join(md) + "\n"
    (out_dir / "label_report.md").write_text(text)
    print(text)
    _log(f"wrote {out_dir / 'label_report.md'}")


if __name__ == "__main__":
    main()
