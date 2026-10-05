#!/usr/bin/env python3
"""
doctor.py
=========

Tiny diagnostic. Run BEFORE the full pipeline if something looks frozen.

It checks:
    1. Python version + interpreter
    2. That every pipeline module imports
    3. That the localities CSV exists and has lat/lon
    4. That .env is loadable and EOWEB_USER/PASS are set
    5. That the EOWEB STAC root responds within 10 seconds

Usage:
    python doctor.py --mineral magnesite

If this completes in a few seconds and prints all OK, your environment is
fine and the pipeline should run.  If a check hangs, the printout will tell
you exactly which one.
"""
from __future__ import annotations

import sys
import time
import argparse
import faulthandler
from datetime import datetime

faulthandler.enable()


def step(label: str):
    sys.stderr.write(f"[doctor {datetime.now():%H:%M:%S}] {label} ... ")
    sys.stderr.flush()
    return time.monotonic()


def ok(t0: float, extra: str = "") -> None:
    sys.stderr.write(f"OK ({time.monotonic()-t0:.2f}s){' ' + extra if extra else ''}\n")
    sys.stderr.flush()


def fail(t0: float, msg: str) -> None:
    sys.stderr.write(f"FAIL ({time.monotonic()-t0:.2f}s) {msg}\n")
    sys.stderr.flush()
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mineral", default=None, help="Optional: also check data/<mineral>/localities.csv.")
    args = ap.parse_args()

    # 1) python
    sys.stderr.write(f"[doctor] python = {sys.executable}\n")
    sys.stderr.write(f"[doctor] version = {sys.version.split()[0]}\n")
    sys.stderr.flush()

    # 2) imports
    t0 = step("importing pipeline_utils")
    try:
        import pipeline_utils as pu  # noqa: F401
    except Exception as e:
        fail(t0, repr(e))
    ok(t0)

    for mod in ("pandas", "numpy", "requests", "bs4", "xarray", "rioxarray", "pyproj"):
        t0 = step(f"importing {mod}")
        try:
            __import__(mod)
        except Exception as e:
            fail(t0, repr(e))
        ok(t0)

    # 3) localities csv
    if args.mineral:
        t0 = step(f"loading localities for {args.mineral}")
        try:
            df = pu.load_localities(args.mineral)
        except Exception as e:
            fail(t0, repr(e))
        ok(t0, f"-> {len(df)} rows  (sample: {df.iloc[0]['Name']!r} @ {df.iloc[0]['lat']:.4f},{df.iloc[0]['lon']:.4f})")

    # 4) .env
    t0 = step("loading .env")
    try:
        pu.load_dotenv()
    except Exception as e:
        fail(t0, repr(e))
    ok(t0)

    import os
    # Backward/alternate env var names supported:
    # - ENMAP_USERNAME / ENMAP_PASSWORD (preferred by some setups)
    # - EOWEB_USER / EOWEB_PASS (original)
    user = os.environ.get("EOWEB_USER") or os.environ.get("ENMAP_USERNAME")
    pwd = os.environ.get("EOWEB_PASS") or os.environ.get("ENMAP_PASSWORD")
    base_url = os.environ.get("EOWEB_BASE_URL", "https://geoservice.dlr.de/eoc/ogc/stac/v1")
    if not user or not pwd:
        sys.stderr.write(
            "[doctor] WARN  Missing credentials in .env. Set either "
            "ENMAP_USERNAME/ENMAP_PASSWORD or EOWEB_USER/EOWEB_PASS.\n"
        )
    else:
        sys.stderr.write(f"[doctor] user = {user}\n")

    # 5) STAC root probe
    t0 = step(f"GET {base_url}  (10s timeout)")
    try:
        import requests
        r = requests.get(base_url, auth=((user, pwd) if user and pwd else None), timeout=10)
        ok(t0, f"-> HTTP {r.status_code}, {len(r.content)} bytes")
        if r.status_code == 401:
            sys.stderr.write("[doctor] HINT  401 Unauthorized -- credentials look wrong.\n")
        elif r.status_code >= 400:
            sys.stderr.write(f"[doctor] HINT  HTTP {r.status_code}: {r.text[:200]!r}\n")
    except Exception as e:
        fail(t0, repr(e))

    sys.stderr.write("[doctor] all checks passed.\n")


if __name__ == "__main__":
    main()
