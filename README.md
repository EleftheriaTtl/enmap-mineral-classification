# EnMAP mineral classification dataset

A multi-label hyperspectral dataset for mineral classification: one 64×64
EnMAP SWIR chip per Mindat mine locality, labelled with the minerals recorded
there. Optionally, a segmentation subset for the chips whose mining ground
OpenStreetMap has mapped.

Three steps. One script per step, one notebook per script.

```
code/      01_mindat_dataset.py   02_enmap_chips.py   03_osm_segmentation.py
notebooks/ 01_mindat_dataset.ipynb 02_enmap_chips.ipynb 03_osm_segmentation.ipynb
data/      raw/ sites.csv labels/ interim/ chips/ seg/ reports/
legacy/    the earlier scripts, kept for reference and not imported
```

The scripts do the work and write files. The notebooks only read those files —
they never fetch, never rebuild, and are safe to re-run at any time. Every
stage writes `data/reports/<name>.json` and `.md`, which is what the notebooks
read.

## The three steps

### 1. The point set — `code/01_mindat_dataset.py`

Everything Mindat has, cleaned into one table with a mineral list per point.

```bash
python code/01_mindat_dataset.py --stage all
```

| Stage | What it does |
|---|---|
| `dump` | pull the Mindat API into `data/raw/`; skipped if the dump is there |
| `clean` | → `data/sites.csv`: drop points with no coordinates, drop the `(0,0)` placeholder, drop points with no minerals, merge localities at the same position, annotate locality type and an `is_mine` flag |
| `labels` | count minerals, sweep the vocabulary threshold → `data/labels/` |

Two deliberate decisions: **the mineral list is the label**, so a point without
one is dropped; and **locality type filters nothing** — a point typed
"mountain" can carry a real mineral list, so the type is recorded as a flag
instead. Positions are merged because a 64 px chip at 30 m spans ~1.9 km: two
localities a metre apart are the same pixels, and keeping both would put
identical imagery under two labels and the same ground in two splits.

### 2. The labelled dataset — `code/02_enmap_chips.py`

For each site, find the EnMAP scene covering it, cut 64×64, save it with its
label.

```bash
python code/02_enmap_chips.py --stage footprints      # ~1 h, once
python code/02_enmap_chips.py --stage match
python code/02_enmap_chips.py --stage chips --limit 2 --verbose   # try it
for i in 0 1 2 3; do tmux new -d -s c$i \
  "python code/02_enmap_chips.py --stage chips --shard $i/4 2>&1 | tee -a chips$i.log"; done
python code/02_enmap_chips.py --stage index
```

| Stage | What it does |
|---|---|
| `footprints` | harvest every EnMAP L2A footprint from the DLR STAC catalogue → `data/interim/footprints.csv.gz`. Resumable by 10° tile |
| `match` | spatial join sites × footprints, then pick which single scene each site is cut from → `data/interim/site_scene.csv` |
| `chips` | read the scenes' COGs over HTTP range requests, cut the windows → `data/chips/<scene>/<site>.npz`. Resumable, shardable |
| `index` | → `data/chips/index.csv`, one row per chip; recount the vocabulary over what was built |

**Why range reads.** The assets are Cloud-Optimized GeoTIFFs in 512×512
pixel-interleaved tiles, so one tile carries every band: the first chip in a
tile costs ~50 MB and the rest are free from GDAL's cache. Downloading the
archives instead would be ~6.6 TB. Asset URLs are derived from the scene id, so
the catalogue is not needed here — which matters, because the catalogue host
blocks some networks while the download host does not.

**Season.** Vegetation hides the mineral signal in SWIR, so scenes are
preferred from the local dry window — June–September north of the equator,
December–March south of it, unconstrained within 15° of it. `--season strict`
refuses out-of-season scenes (and loses the sites that have none);
`--season off` ignores season entirely.

**One scene per site.** Scene choice is a greedy cover restricted to in-season
scenes, and each site is then assigned a scene from within that chosen set.
Choosing per site independently would multiply the scenes to read without
adding a single chip.

**Masking is not baked in.** `X` is stored exactly as the sensor gave it. All
five EnMAP quality layers are stored raw per chip (`qmask`), and `valid` is
derived from whichever of them `--mask-layers` names (default
`cloud,cloudshadow`). Changing that decision later costs one line at training
time, not a rebuild. Cloud is measured over the 64×64 patch, not the 30 km
scene.

Each chip is one `.npz`:

| Key | What it is |
|---|---|
| `X` | int16, (bands, 64, 64), reflectance × 1e-4 |
| `valid` | uint8, 1 where the pixel is usable |
| `qmask`, `qmask_names`, `patch_frac` | the five quality layers, raw, and how much of the patch each flags |
| `wavelength` | nm per band; the 1330–1450 nm water-vapour gap is dropped |
| `minerals`, `mineral_ids` | the label, `;`-separated |
| `crs_wkt`, `transform`, `chip_px` | the window's grid, so a mask can be rasterized onto these exact pixels |
| `patch_cloud`, `lat`, `lon`, `scene_id`, `is_mine` | provenance and filters |

### 3. The segmentation subset — `code/03_osm_segmentation.py` (optional)

Keep the chips where OSM maps mining ground, and rasterize those polygons onto
the chip's own grid as the target mask.

```bash
python code/03_osm_segmentation.py --stage query --limit 50   # try it
python code/03_osm_segmentation.py --stage query
python code/03_osm_segmentation.py --stage masks
python code/03_osm_segmentation.py --stage index
```

Queries run one per 0.25° cell rather than per chip, because Overpass is the
bottleneck and a cell usually holds several chips; cached cells are never
queried twice. A chip without a mask means nobody mapped that pit, not that
there is no pit — the masks are positive-only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then fill it in
```

`.env` needs `ENMAP_USERNAME` / `ENMAP_PASSWORD` (DLR EOC account, for every
EnMAP read) and `MINDAT_API_KEY` (only to refresh the Mindat dump).

## Conventions

- Paths come from `code/common.py`; nothing hard-codes a directory.
- A list in a CSV cell is `;`-separated.
- Every stage is resumable and skips work already on disk, so a killed run is
  restarted, never restarted from scratch.
- Reports are written as both JSON and Markdown: the JSON is what the notebooks
  plot, the Markdown is what gets pasted into an email.
