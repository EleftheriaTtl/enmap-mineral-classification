#!/usr/bin/env python3
"""Generate 10_explore_localities.ipynb (Phase 0 exploration notebook)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Mindat localities — what is actually in the raw data

Phase 0, from the beginning. Nothing is filtered here: this notebook reads the raw Mindat
dump and shows what the locality types are, so the mine-related filter is a decision you
make with the numbers in front of you rather than a regex someone wrote earlier.

Inputs (both already on the VM):

- `data/_mindat/localities.jsonl.gz` — 451,221 localities, one JSON object per line
- `data/_mindat/locality_types.csv` — `lt_id` -> type name

Run top to bottom. The load cell takes a minute or two.
""")

code("""
from pathlib import Path
import gzip, json, re
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

RAW = Path("data/_mindat/localities.jsonl.gz")
TYPES = Path("data/_mindat/locality_types.csv")
SITES = Path("data/sites.csv")          # what the earlier build produced, for comparison

for p in (RAW, TYPES):
    print(f"{'OK ' if p.exists() else 'MISSING'} {p}")
""")

md("## 1. The type lookup")

code("""
types = pd.read_csv(TYPES)
print(types.shape, list(types.columns))
types.head(10)
""")

md("""
## 2. Load the localities

One pass over the file, keeping only the fields Phase 0 needs. `geomaterials` is the
mineral list per locality — kept as a count here, and as the raw list for later label work.
""")

code("""
KEEP = ("id", "locality_type", "txt", "latitude", "longitude", "country",
        "non_hierarchical", "loc_status", "level", "parent", "elements")

def load(path=RAW, limit=None):
    rows = []
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            r = json.loads(line)
            row = {k: r.get(k) for k in KEEP}
            gm = r.get("geomaterials") or []
            row["n_geomaterials"] = len(gm) if isinstance(gm, (list, tuple)) else 0
            rows.append(row)
    return pd.DataFrame(rows)

loc = load()
print(f"{len(loc):,} localities, {loc.shape[1]} columns")
loc.head(3)
""")

md("""
## 3. Coordinates

A locality with no usable coordinate cannot be imaged, whatever its type. (0, 0) is Mindat's
placeholder for unknown, not a real position in the Gulf of Guinea.
""")

code("""
lat = pd.to_numeric(loc["latitude"], errors="coerce")
lon = pd.to_numeric(loc["longitude"], errors="coerce")
loc["lat"], loc["lon"] = lat, lon
loc["has_xy"] = (lat.notna() & lon.notna()
                 & lat.between(-90, 90) & lon.between(-180, 180)
                 & ~((lat == 0) & (lon == 0)))

print(f"with usable coordinates : {loc.has_xy.sum():,}  ({100*loc.has_xy.mean():.1f}%)")
print(f"missing or out of range : {(~loc.has_xy).sum():,}")
print(f"exactly (0, 0)          : {int(((lat == 0) & (lon == 0)).sum()):,}")
""")

md("""
### Coordinate precision

Decimal places are a rough proxy for how precisely a locality is placed. A value given to
2 decimals is a ~1 km box — larger than the mine, but still inside a 3.8 km chip.
""")

code("""
def decimals(v):
    s = f"{v:.8f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0

sample = loc.loc[loc.has_xy, ["lat", "lon"]]
prec = sample.lat.map(decimals).combine(sample.lon.map(decimals), max)
print(prec.value_counts().sort_index().to_string())
print(f"\\n<= 2 decimals (~1 km or coarser): {(prec <= 2).sum():,} "
      f"({100*(prec <= 2).mean():.1f}% of located localities)")
""")

md("""
## 4. The type table

Every type, how many localities carry it, and how many of those have coordinates. This is
the table the filter decision should be made from.
""")

code("""
name_of = types.set_index("lt_id")["lt_text"].to_dict()

tt = (loc.groupby("locality_type", dropna=False)
        .agg(n=("id", "size"), with_xy=("has_xy", "sum"))
        .reset_index())
tt["name"] = tt.locality_type.map(lambda i: name_of.get(i, "(unset)" if not i else f"(unknown id {i})"))
tt["pct_of_all"] = 100 * tt.n / len(loc)
tt = tt.sort_values("with_xy", ascending=False).reset_index(drop=True)

print(f"{len(tt)} distinct types; top 40 by located count\\n")
tt.head(40)[["name", "n", "with_xy", "pct_of_all"]]
""")

code("""
# the long tail
print(f"types with >= 1,000 located localities : {(tt.with_xy >= 1000).sum()}")
print(f"types with < 100 located localities    : {(tt.with_xy < 100).sum()}")
print(f"localities in those small types        : {tt.loc[tt.with_xy < 100, 'with_xy'].sum():,}")
""")

md("""
## 5. What the old regex keeps

This is the pattern in `00b_filter_localities.py`. Run it against the type names to see what
it would keep and, more importantly, what it drops.
""")

code("""
OLD_RX = re.compile(
    r"\\b(?:mines?|quarr(?:y|ies)|pits?|open[- ]?(?:cut|pit|cast)|deposits?|prospects?|"
    r"occurrences?|adits?|shafts?|claims?|workings?|trench(?:es)?|dumps?|tailings|"
    r"group of mines|mining area|orebody|ore body|lode|vein)\\b", re.I)

tt["old_regex"] = tt.name.map(lambda s: bool(OLD_RX.search(str(s))))

kept = int(tt.loc[tt.old_regex, "with_xy"].sum())
print(f"kept by the old regex (with coords): {kept:,}")
print(f"dropped                            : {int(tt.with_xy.sum()) - kept:,}\\n")
print("Largest types it DROPS:")
tt[~tt.old_regex].head(20)[["name", "with_xy"]]
""")

md("""
Look closely at the dropped list. Expected to show up there:

- **Colliery** — a coal mine. The regex has no word for it.
- **(unset)** — 70,026 localities with no type at all.
- **Mining District, Mining Area** — aggregates, excluded on purpose; they are regions, not sites.
- **Placer, Pegmatite, Outcrop, Property** — genuine judgment calls.
""")

md("""
## 6. Build your own keep-list

Edit `KEEP_TYPES` / `DROP_TYPES` and re-run. The cell reports the resulting site count so you
can see the cost of each decision immediately.
""")

code("""
KEEP_TYPES = [
    "Mine", "Quarry", "Prospect", "Deposit", "Occurrence", "Pit", "Claim",
    "Colliery", "Shaft", "Adit", "Open Cut", "Open Pit", "Working", "Workings",
    "Vein", "Lode", "Orebody", "Group of Mines", "Placer",
]
DROP_TYPES = ["Mining District", "District", "Mining Area", "Area", "Region"]

keep_set = {s.lower() for s in KEEP_TYPES} - {s.lower() for s in DROP_TYPES}
tt["mine_like"] = tt.name.str.lower().isin(keep_set)

chosen = int(tt.loc[tt.mine_like, "with_xy"].sum())
print(f"localities kept: {chosen:,}   (old regex: {int(tt.loc[tt.old_regex, 'with_xy'].sum()):,})")
print("\\nkept types:")
print(tt[tt.mine_like][["name", "with_xy"]].to_string(index=False))
""")

code("""
# apply it to the localities themselves
keep_ids = set(tt.loc[tt.mine_like, "locality_type"])
mine = loc[loc.has_xy & loc.locality_type.isin(keep_ids)].copy()
print(f"{len(mine):,} located, mine-like localities")

if "non_hierarchical" in mine.columns:
    nh = pd.to_numeric(mine.non_hierarchical, errors="coerce").fillna(0).astype(int) == 1
    print(f"  of which non-hierarchical overlays: {int(nh.sum()):,}")
""")

md("""
## 7. Deduplicate by position

Two questions, kept separate:

- **identical coordinates** — the same mine entered twice; always merge
- **within a chip** — different mines whose 3.8 km chips would overlap; a modelling decision
""")

code("""
for dp, about in [(5, "~1 m"), (4, "~11 m"), (3, "~110 m"), (2, "~1.1 km")]:
    key = mine.lat.round(dp).astype(str) + "," + mine.lon.round(dp).astype(str)
    print(f"round to {dp} dp ({about:>7}): {key.nunique():,} distinct sites "
          f"(-{len(mine) - key.nunique():,})")
""")

code("""
# a 128 x 128 chip at 30 m spans 3.84 km: how many sites share one?
CHIP_M = 128 * 30

def grid_key(df, metres):
    dlat = metres / 111_320.0
    dlon = dlat / np.maximum(np.cos(np.radians(df.lat)), 0.01)
    return (np.floor(df.lat / dlat).astype(int).astype(str) + "_"
            + np.floor(df.lon / dlon).astype(int).astype(str))

for m in (CHIP_M / 3, CHIP_M):
    k = grid_key(mine, m)
    print(f"one site per {m/1000:.2f} km cell: {k.nunique():,} sites "
          f"({100*k.nunique()/len(mine):.1f}% of {len(mine):,})")
""")

md("""
## 8. Compare with the existing sites.csv

Does the earlier build agree with what this notebook produces? A large gap means the old
filter made choices worth revisiting.
""")

code("""
if SITES.exists():
    sites = pd.read_csv(SITES, low_memory=False)
    print(f"sites.csv: {len(sites):,} rows, {sites.shape[1]} columns")
    print(list(sites.columns))
    print(f"\\nthis notebook, mine-like + located : {len(mine):,}")
    print(f"sites.csv                          : {len(sites):,}")
else:
    print("no sites.csv yet")
""")

md("""
## 9. Write the locality table for the next step

Only when the numbers above look right. This is the input the STAC coverage step reads.
""")

code("""
OUT = Path("data/localities_mine_like.csv")

def write_sites(df, dedupe_dp=5):
    key = df.lat.round(dedupe_dp).astype(str) + "," + df.lon.round(dedupe_dp).astype(str)
    out = df.assign(coord_key=key).drop_duplicates("coord_key").copy()
    out["type_name"] = out.locality_type.map(lambda i: name_of.get(i, ""))
    out = out.rename(columns={"id": "site_id", "txt": "name"})
    cols = ["site_id", "name", "type_name", "lat", "lon", "country",
            "n_geomaterials", "elements", "coord_key"]
    out[[c for c in cols if c in out.columns]].to_csv(OUT, index=False)
    return out

# written = write_sites(mine)
# print(f"{len(written):,} sites -> {OUT}")
print("uncomment the two lines above once the filter is settled")
""")

nb["cells"] = C
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.write(nb, "10_explore_localities.ipynb")
print("wrote 10_explore_localities.ipynb")
