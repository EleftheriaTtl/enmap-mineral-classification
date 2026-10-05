"""
common.py
=========

Everything the three steps share: where files live, logging, credentials, and
the EnMAP URL / band arithmetic. Nothing in here does work of its own.

Layout it defines (all paths relative to the repository root):

    data/
      raw/            Mindat dumps, exactly as the API returned them
      sites.csv       step 01 output: the clean, labelled point set
      labels/         step 01 output: mineral counts and the label vocabulary
      interim/        step 02 working tables: footprints, scene assignment
      chips/          step 02 output: the labelled chip dataset + index.csv
      seg/            step 03 output: the segmentation subset + index.csv
      reports/        one .json and one .md per stage, for the notebooks
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
LABELS = DATA / "labels"
CHIPS = DATA / "chips"
SEG = DATA / "seg"
REPORTS = DATA / "reports"

SITES_CSV = DATA / "sites.csv"
LOCALITIES_JSONL = RAW / "localities.jsonl.gz"
LOCALITY_TYPES_CSV = RAW / "locality_types.csv"
GEOMATERIALS_CSV = RAW / "geomaterials.csv"
FOOTPRINTS_CSV = INTERIM / "footprints.csv.gz"
SITE_SCENE_CSV = INTERIM / "site_scene.csv"
SCENE_PRIORITY_CSV = INTERIM / "scene_priority.csv"
CHIP_INDEX_CSV = CHIPS / "index.csv"
SEG_INDEX_CSV = SEG / "index.csv"


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, LABELS, CHIPS, SEG, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# logging / reports
# --------------------------------------------------------------------------
def log(tag: str, msg: str) -> None:
    print(f"[{tag} {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def write_report(name: str, stats: dict, markdown: str) -> None:
    """Write data/reports/<name>.json and .md. The notebooks read these."""
    import json

    REPORTS.mkdir(parents=True, exist_ok=True)
    stats = {**stats, "generated": datetime.now().isoformat(timespec="seconds")}
    (REPORTS / f"{name}.json").write_text(json.dumps(stats, indent=2, default=str))
    (REPORTS / f"{name}.md").write_text(markdown)
    log("report", f"wrote reports/{name}.json and reports/{name}.md")


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env reader; existing environment variables win."""
    p = Path(path) if path else (REPO / ".env")
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_any(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def require_env_any(*names: str) -> str:
    v = env_any(*names)
    if not v:
        sys.exit(f"none of {', '.join(names)} is set; add one to .env")
    return v


def dlr_auth() -> tuple[str, str]:
    """DLR EOC / geoservice credentials, used for every EnMAP read."""
    load_dotenv()
    return (require_env_any("ENMAP_USERNAME", "EOWEB_USER"),
            require_env_any("ENMAP_PASSWORD", "EOWEB_PASS"))


# --------------------------------------------------------------------------
# EnMAP asset URLs, derived from the scene id alone
# --------------------------------------------------------------------------
DOWNLOAD_BASE = "https://download.geoservice.dlr.de/ENMAP/files/L2A"
STAC_BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION = "ENMAP_HSI_L2A"

NODATA = -9999
PIXEL_M = 30.0

QUALITY_ASSETS = ("QL_QUALITY_CLOUD", "QL_QUALITY_CLOUDSHADOW", "QL_QUALITY_HAZE",
                  "QL_QUALITY_CIRRUS", "QL_QUALITY_SNOW")
SHORT = {"cloud": "QL_QUALITY_CLOUD", "cloudshadow": "QL_QUALITY_CLOUDSHADOW",
         "haze": "QL_QUALITY_HAZE", "cirrus": "QL_QUALITY_CIRRUS",
         "snow": "QL_QUALITY_SNOW"}
LONG = {v: k for k, v in SHORT.items()}

SCENE_RE = re.compile(r"^(ENMAP01-____L2A-(DT\d+)_(\d{4})(\d{2})(\d{2})T\d{6}Z_(\d{3})_.*)$")


def scene_urls(scene_id: str) -> dict[str, str]:
    """
    ENMAP01-____L2A-DT0000124671_20250416T003107Z_003_V010502_20250418T141036Z
      -> .../L2A/2025/04/16/DT0000124671/03/<scene_id>/<scene_id>-<ASSET>

    The STAC catalogue is never needed to find an asset, which matters because
    the catalogue host blocks some networks while the download host does not.
    """
    m = SCENE_RE.match(scene_id)
    if not m:
        raise ValueError(f"unrecognised scene id: {scene_id}")
    full, dt, yyyy, mm, dd, counter = m.groups()
    folder = f"{DOWNLOAD_BASE}/{yyyy}/{mm}/{dd}/{dt}/{int(counter):02d}/{full}/{full}"
    out = {"metadata": f"{folder}-METADATA.XML",
           "image": f"{folder}-SPECTRAL_IMAGE_COG.TIF"}
    for a in QUALITY_ASSETS + ("QL_PIXELMASK",):
        out[a] = f"{folder}-{a}_COG.TIF"
    return out


def with_auth(url: str) -> str:
    """Inline credentials, which is how GDAL's /vsicurl/ accepts basic auth."""
    u, p = dlr_auth()
    return url.replace("https://", f"https://{u}:{p}@")


def scene_date(scene_id: str):
    m = SCENE_RE.match(scene_id)
    if not m:
        return None
    _, _, y, mo, d, _ = m.groups()
    return datetime(int(y), int(mo), int(d))


def gdal_env(cache_mb: int = 2048) -> None:
    """Settings that make range reads over HTTP fast and quiet."""
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.XML")
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
    os.environ.setdefault("GDAL_HTTP_VERSION", "2")
    os.environ["GDAL_CACHEMAX"] = str(cache_mb)


# --------------------------------------------------------------------------
# band selection from a scene's own metadata
# --------------------------------------------------------------------------
SWIR_GAP = (1330.86, 1449.43)      # the water-vapour band dropped by version 128


def band_plan(meta_url: str, version: str = "128") -> tuple[np.ndarray, np.ndarray]:
    """
    Return (1-based band indices for rasterio, wavelengths) for the SWIR set.

    version: "128" drops the water-vapour gap, "58" keeps >= 1967 nm only,
    "v1" keeps every SWIR band the metadata declares.
    """
    import xml.etree.ElementTree as ET

    import httpx

    u, p = dlr_auth()
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        r = c.get(meta_url, auth=(u, p))
        r.raise_for_status()
        root = ET.fromstring(r.content)

    strip = lambda t: t.split("}")[-1]

    centres: list[float] = []
    for band in root.iter():
        if strip(band.tag) != "bandID":
            continue
        for child in band:
            if strip(child.tag) == "wavelengthCenterOfBand" and child.text:
                centres.append(float(child.text))
                break
    wl = np.asarray(centres, dtype=np.float32)
    if wl.size == 0:
        raise ValueError("no wavelengths in metadata")

    swir: list[float] = []
    for el in root.iter():
        if strip(el.tag) != "SWIR":
            continue
        for sub in el.iter():
            if strip(sub.tag) == "wavelength" and sub.text:
                swir.append(float(sub.text))
    if swir:
        sw = np.asarray(swir, dtype=np.float32)
        keep = np.array(sorted({int(np.argmin(np.abs(wl - w))) for w in sw}), dtype=int)
    else:
        keep = np.where(wl >= 900.0)[0]

    if version == "128":
        keep = keep[~((wl[keep] > SWIR_GAP[0]) & (wl[keep] < SWIR_GAP[1]))]
    elif version == "58":
        keep = keep[wl[keep] >= 1967.66]

    keep = keep[np.argsort(wl[keep])]
    return keep + 1, wl[keep]


_TX: dict = {}


def to_scene_xy(crs, lon: float, lat: float) -> tuple[float, float]:
    """Lon/lat -> the scene's projected CRS, with the transformer cached."""
    from pyproj import Transformer

    key = str(crs)
    if key not in _TX:
        _TX[key] = Transformer.from_crs("EPSG:4326", key, always_xy=True)
    return _TX[key].transform(lon, lat)


# --------------------------------------------------------------------------
# small table helpers
# --------------------------------------------------------------------------
def split_list(v, sep: Optional[str] = None) -> list[str]:
    """
    Parse one cell that holds a list: 'a;b;c', "['a', 'b']", 'a|b', 'a,b'.
    Mindat and our own CSVs disagree about the separator, so detect it.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    if s.startswith("[") and s.endswith("]"):
        found = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
        if found:
            return [a or b for a, b in found]
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


def join_list(values: Iterable) -> str:
    """Our canonical on-disk form for a list column."""
    return ";".join(str(v) for v in values)


def md_table(header: list[str], rows: Iterable[Iterable], align: str = "") -> list[str]:
    """Markdown table lines. `align` is one char per column: 'l' or 'r'."""
    align = align or "l" * len(header)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return out
