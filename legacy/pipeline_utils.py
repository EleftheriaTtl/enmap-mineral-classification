"""
pipeline_utils.py
=================

Shared helpers for the EnMAP mineral-classification pipeline.

All numbered scripts (00_..04_) and run_pipeline.py import from here so the
on-disk layout stays consistent:

    data/
      <mineral>/
        localities.csv          (00 output)
        stac_matches.csv        (01 output)
        raw/                    (01 output: downloaded EnMAP zips)
        netcdf/                 (02 output: full-scene NetCDFs)
        chips/<mine>/           (03 output: clipped NetCDF + RGB + npy)
        masks/<mine>/           (04 output: OSM mine/quarry masks)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Repository root = directory of this file.
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def normalize_mineral(name: str) -> str:
    """Lower-case, strip whitespace; used for folder names."""
    return str(name).strip().lower()


def safe_name(s: str, max_len: int = 80) -> str:
    """Filesystem-safe slug for a mine name."""
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:max_len] if len(s) > max_len else s


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def mineral_dir(mineral: str, *, create: bool = True) -> Path:
    p = DATA_ROOT / normalize_mineral(mineral)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def localities_csv(mineral: str) -> Path:
    return mineral_dir(mineral) / "localities.csv"


def stac_matches_csv(mineral: str) -> Path:
    return mineral_dir(mineral) / "stac_matches.csv"


def raw_dir(mineral: str, *, create: bool = True) -> Path:
    p = mineral_dir(mineral) / "raw"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def netcdf_dir(mineral: str, *, create: bool = True) -> Path:
    p = mineral_dir(mineral) / "netcdf"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def chips_dir(mineral: str, *, create: bool = True) -> Path:
    p = mineral_dir(mineral) / "chips"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def masks_dir(mineral: str, *, create: bool = True) -> Path:
    p = mineral_dir(mineral) / "masks"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def mine_chip_dir(mineral: str, mine_name: str, *, create: bool = True) -> Path:
    p = chips_dir(mineral) / safe_name(mine_name)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def mine_mask_dir(mineral: str, mine_name: str, *, create: bool = True) -> Path:
    p = masks_dir(mineral) / safe_name(mine_name)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_localities(mineral_or_csv: str) -> pd.DataFrame:
    """
    Load a localities CSV. Accepts either a mineral name (resolves to
    data/<mineral>/localities.csv) or an explicit path.

    Returned DataFrame is guaranteed to have columns: Name, lat, lon.
    Other columns are preserved.
    """
    p = Path(mineral_or_csv)
    if not p.exists():
        # treat as mineral name
        p = localities_csv(mineral_or_csv)
    if not p.exists():
        raise FileNotFoundError(f"Localities CSV not found: {p}")

    df = pd.read_csv(p)

    # normalize lat/lon
    if "lat" not in df.columns or "lon" not in df.columns:
        if "Latitude & Longitude" in df.columns:
            lats, lons = [], []
            for v in df["Latitude & Longitude"]:
                try:
                    a, b = str(v).split(",", 1)
                    lats.append(float(a))
                    lons.append(float(b))
                except Exception:
                    lats.append(float("nan"))
                    lons.append(float("nan"))
            df["lat"] = lats
            df["lon"] = lons
        else:
            raise ValueError(
                f"{p} must have lat/lon columns or a 'Latitude & Longitude' column."
            )

    if "Name" not in df.columns:
        # fall back to a generated name
        df["Name"] = [f"site_{i:05d}" for i in range(len(df))]

    df = df.dropna(subset=["lat", "lon"]).copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
def load_dotenv(path: Optional[Path] = None) -> None:
    """
    Tiny stand-alone .env loader (so we don't strictly require python-dotenv).
    Sets os.environ for any KEY=VALUE lines that aren't comments and aren't
    already set in the environment.
    """
    p = Path(path) if path else (REPO_ROOT / ".env")
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Add it to your .env (see .env.example) or export it."
        )
    return val


def require_env_any(*names: str) -> str:
    """
    Return the first set environment variable among `names`.

    Useful for backward-compatible renames, e.g. ENMAP_USERNAME vs EOWEB_USER.
    """
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(
        "None of these environment variables are set: "
        + ", ".join(names)
        + ". Add one of them to your .env (or export it)."
    )
