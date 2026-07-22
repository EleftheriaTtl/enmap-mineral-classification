# enmap-mineral-classification

End-to-end pipeline to build a **segmentation-ready dataset** from EnMAP L2A
hyperspectral scenes around mine localities for a given mineral:

- Find mine localities (lat/lon) for a mineral
- Search EnMAP L2A STAC and download scenes
- Convert each scene into a georeferenced reflectance cube (NetCDF)
- Clip per-mine chips
- Create OpenStreetMap mine/quarry masks aligned to each chip
- Package `(X, y, wavelength, metadata)` into ML-ready `.npz` samples

## What you get (high level)

Given a mineral (e.g. `magnesite`), the pipeline produces a folder:

`data/<mineral>/dataset/`

containing:

- `manifest.csv` (one row per sample)
- `samples/<chip_id>.npz` where each `.npz` includes:
  - `X`: reflectance cube, shape `(B, H, W)` float32
  - `y`: OSM mask, shape `(H, W)` uint8 (0 background, 1 mine/quarry)
  - `wavelength`: shape `(B,)` float32 (nm)
  - plus identifiers like `chip_id`, `mine_name`, `scene_id`, `height`, `width`

## Pipeline steps

```
00_crawl_mindat.py            crawl Mindat for localities of <mineral>
01_stac_search_download.py    STAC search + download EnMAP L2A scenes
02_enmap_process.py           EnMAP archives -> full-scene NetCDF (reflectance + wavelengths)
03_clip_mines.py              clip per-mine chips (buffer in meters or fixed chip_px)
04_osm_mask_batch.py          OSM mine/quarry masks for each chip
05_build_dataset.py           chips + masks -> ML-ready .npz dataset
run_pipeline.py               run the whole pipeline for one mineral
```

## On-disk layout

Everything for one mineral lives under `data/<mineral>/`:

```
data/
  magnesite/
    localities.csv            # 00 output
    stac_matches.csv          # 01 output (1 best scene per mine)
    raw/                      # 01 output: downloaded EnMAP L2A archives
    netcdf/                   # 02 output: full-scene NetCDFs
    chips/<mine>/             # 03 output: clipped NetCDF + RGB png + .npy
    masks/<mine>/             # 04 output: OSM mine/quarry masks (uint8)
    dataset/                  # 05 output: ML-ready dataset
      manifest.csv
      samples/<chip_id>.npz   # X, y, wavelength, scene_id, mine_name, ...
```

`new_enmap.py` (the `EnMAPProcessor` class) stays at the repo root. Old
versions of the scripts and the exploratory notebooks have been moved to
`legacy/` and the previously hand-crawled CSVs are in `legacy/sample_csvs/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env -> ENMAP_USERNAME / ENMAP_PASSWORD
```

### Authentication (step 01 downloads)

Step 01 downloads EnMAP assets that may require DLR EOC SSO. Credentials are read
from `.env`:

```
ENMAP_USERNAME = your_username
ENMAP_PASSWORD = your_password
```

## Run the whole pipeline

```bash
python run_pipeline.py --mineral magnesite
```

Useful overrides:

```bash
# Smaller chip
python run_pipeline.py --mineral magnesite --buffer-m 750

# Stricter cloud filter, last 2 years only
python run_pipeline.py --mineral galena --cloud-max 5 \
    --start 2024-01-01 --end 2025-12-31

# Skip earlier steps if their outputs already exist
python run_pipeline.py --mineral cuprite --skip-step 00 --skip-step 01

# Use the legacy fixed-pixel chip behavior instead of a meter buffer
python run_pipeline.py --mineral magnesite --chip-px 128 --chip-size 128
```

### Running in tmux

If you want the run to survive disconnects:

```bash
tmux new -s magnesite
source .venv/bin/activate
python run_pipeline.py --mineral magnesite --skip-step 00
```

## Run individual steps

Each script is independently runnable:

```bash
python 00_crawl_mindat.py --mineral magnesite --resume
python 01_stac_search_download.py --mineral magnesite --cloud-max 10
python 02_enmap_process.py --mineral magnesite --keep-only-swir --version 128 --apply-cloud-mask
python 03_clip_mines.py --mineral magnesite --chip-px 128
python 04_osm_mask_batch.py --mineral magnesite
python 05_build_dataset.py --mineral magnesite --chip-size 128
```

## Use an existing localities CSV (skip the crawl)

If you already have a CSV with `Name`, `lat`, `lon` columns (the included
`legacy/sample_csvs/magnesite_mines_with_names.csv` does), just drop it at
the canonical path and skip step 00:

```bash
mkdir -p data/magnesite
cp legacy/sample_csvs/magnesite_mines_with_names.csv data/magnesite/localities.csv

python run_pipeline.py --mineral magnesite \
    --skip-step 00 \
    --chip-px 128 --chip-size 128 \
    --keep-only-swir --version 128 --apply-cloud-mask
```

`--chip-px 128` makes step 03 produce 128x128 chips (instead of the meter
buffer); `--chip-size 128` makes step 05 enforce that size when packaging.

## Results / Example

The notebook `06_showcase_dataset.ipynb` demonstrates the end result on **one**
sample from `data/magnesite/dataset/`:

- Loads `manifest.csv`
- Opens one `.npz` sample (reflectance cube `X`, mask `y`, wavelengths)
- Plots:
  - an RGB composite built from bands nearest ~650/560/490 nm
  - the OSM mask
  - an overlay (mask on top of RGB)

In the example run shown in the notebook, samples are:

- `X`: `(224, 100, 100)` (224 spectral bands; 100×100 spatial chip)
- `y`: `(100, 100)` (0/1 mask)

## Loading the dataset

Each `.npz` is self-contained:

```python
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

manifest = pd.read_csv("data/magnesite/dataset/manifest.csv")

class EnMAPSegDataset(Dataset):
    def __init__(self, manifest_df):
        self.rows = manifest_df.reset_index(drop=True)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        z = np.load(r["npz_path"])
        X = torch.from_numpy(z["X"])              # (B, H, W) float32
        y = torch.from_numpy(z["y"]).long()       # (H, W)    0/1
        return X, y

loader = DataLoader(EnMAPSegDataset(manifest), batch_size=8, shuffle=True)
```

The manifest's `split` column starts as `unassigned`; fill it in from your
training code (e.g. group by `mine_name` to avoid leakage).

Steps 02/03/04 also support a `--base-dir` / `--legacy-out-dir` mode that
mirrors the old standalone-script behavior, for backward compatibility with
existing data on external drives.

## Notes

- The Mindat crawler is **HTML scraping**: be polite (`--sleep`), respect
  Mindat's terms of use, and prefer the on-disk cache (`./cache`) on re-runs.
- Step 01 downloads **one best scene per mine** (lowest cloud cover, then most
  recent), and enforces cloud/quality filters client-side based on STAC item
  properties. Tweak `01_stac_search_download.py:best_scene` if you want
  different selection logic.
- Step 03 defaults to a **1500 m buffer** around each (lat, lon). The chip is
  square in CRS units (typically meters), so its pixel size depends on the
  scene's GSD (~30 m for EnMAP).
- Step 04 uses OSM tags `landuse=quarry|mine`, `industrial=mining`,
  `man_made=tailings_pond|tailings|spoil_heap`. Edit `DEFAULT_TAGS` in
  `04_osm_mask_batch.py` to extend.
