#!/usr/bin/env bash
# migrate.sh — move an existing checkout to the three-step layout.
#
# Safe to run twice: every move is skipped if it has already happened, and
# nothing is deleted. Run it from the repository root.
#
#   bash migrate.sh
#
# What it does
#   1. moves the old scripts and notebooks into legacy/
#   2. moves existing data into the new data/ layout, so no stage re-runs
#   3. sets aside chips built before the format change, because mixing formats
#      would corrupt data/chips/index.csv
set -uo pipefail
cd "$(dirname "$0")"

say() { printf '  %s\n' "$*"; }

echo "1. old scripts -> legacy/"
mkdir -p legacy
OLD=(00_crawl_mindat.py 01_stac_search_download.py 02_enmap_process.py
     03_clip_mines.py 04_osm_mask_batch.py 05_build_dataset.py
     build_chips.py build_chips_cog.py build_notebook.py diag_masks.py
     doctor.py geoserver_download.py new_enmap.py phase0_budget.py
     phase0_coverage.py phase0_labels.py phase0_season.py pipeline_utils.py
     probe_stac.py run_pipeline.py 06_showcase_dataset.ipynb
     10_explore_localities.ipynb test_enmap_processor.ipynb)
for f in "${OLD[@]}"; do
  [ -e "$f" ] || continue
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    git mv -f "$f" legacy/ && say "git mv $f"
  else
    mv -f "$f" legacy/ && say "mv $f"
  fi
done
rm -rf __pycache__

echo "2. data -> the new layout"
mkdir -p data/raw data/interim data/labels data/chips data/seg data/reports
move() {
  if [ -e "$1" ] && [ ! -e "$2" ]; then
    mkdir -p "$(dirname "$2")"; mv "$1" "$2"; say "$1 -> $2"
  fi
}
# the Mindat dump
move data/_mindat/localities.jsonl.gz data/raw/localities.jsonl.gz
move data/_mindat/locality_types.csv  data/raw/locality_types.csv
move data/_mindat/geomaterials.csv    data/raw/geomaterials.csv
for f in data/_mindat/*; do move "$f" "data/raw/$(basename "$f")"; done
# the cleaned point set
move data/sites_v2.csv     data/sites.csv
move data/sites_v2.parquet data/sites.parquet
# the footprint harvest and its resume state
move data/footprints.csv.gz                 data/interim/footprints.csv.gz
move data/footprints.csv.gz.tiles.json      data/interim/footprints.csv.gz.tiles.json
# the old coverage tables: superseded by `--stage match`, kept for reference
if [ -d data/coverage_v2_c50q2 ] && [ ! -d data/interim/old_coverage_v2_c50q2 ]; then
  mv data/coverage_v2_c50q2 data/interim/old_coverage_v2_c50q2
  say "data/coverage_v2_c50q2 -> data/interim/old_coverage_v2_c50q2"
fi
# the label tables, if the old phase0_labels run produced them
for f in mineral_counts.csv label_vocab.csv label_report.md; do
  move "data/labels_v2/$f" "data/labels/$f"
done

rmdir data/_mindat data/labels_v2 2>/dev/null || true
echo "3. chips built before the format change"
# Chips are only compatible if they carry the window's grid (crs_wkt), which
# the current builder stores and the previous one did not.
OLDCHIPS=$(python3 - <<'PY'
import glob, sys
import numpy as np
f = sorted(glob.glob("data/chips/*/*.npz"))
if not f:
    print("none"); sys.exit()
try:
    d = np.load(f[0], allow_pickle=False)
    print("ok" if "crs_wkt" in d else "old")
except Exception:
    print("old")
PY
)
case "$OLDCHIPS" in
  none) say "no chips yet — nothing to do" ;;
  ok)   say "existing chips carry the new format; left in place" ;;
  old)  if [ ! -d data/_chips_old_format ]; then
          mv data/chips data/_chips_old_format
          mkdir -p data/chips
          say "existing chips predate the format change (no crs_wkt, pixels"
          say "overwritten by the mask) -> data/_chips_old_format"
          say "delete that folder once you are happy, then rebuild."
        else
          say "data/_chips_old_format already exists; left as is"
        fi ;;
esac

echo
echo "done. Next:"
echo "  python code/01_mindat_dataset.py --stage clean    # rebuild sites.csv in place"
echo "  python code/02_enmap_chips.py    --stage match"
echo "  python code/02_enmap_chips.py    --stage chips --limit 2 --verbose"
