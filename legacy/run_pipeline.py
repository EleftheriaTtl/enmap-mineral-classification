#!/usr/bin/env python3
"""
run_pipeline.py
===============

Convenience wrapper that runs all six pipeline steps for a single mineral:

    00_crawl_mindat.py            # crawl localities
    01_stac_search_download.py    # find + download EnMAP scenes
    02_enmap_process.py           # convert L2A archives -> NetCDFs
    03_clip_mines.py              # clip per-mine chips
    04_osm_mask_batch.py          # build OSM mine/quarry masks
    05_build_dataset.py           # pair chips + masks -> ML-ready .npz dataset

Each step is invoked as a subprocess with **unbuffered stdout/stderr** so its
own progress messages appear live. Output is also tee'd to a log file at
`data/<mineral>/pipeline.log` and timing info (start time, end time, elapsed
seconds) is printed before/after each step.

Usage
-----
    python run_pipeline.py --mineral magnesite
    python run_pipeline.py --mineral magnesite --skip-step 00 --chip-px 128 --chip-size 128
"""
from __future__ import annotations

import sys as _sys, time as _time, faulthandler as _fh
_sys.stderr.write(f"[pipeline {_time.strftime('%H:%M:%S')}] booting...\n")
_sys.stderr.flush()
_fh.enable()  # Ctrl+\ dumps the full Python stack to stderr

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import pipeline_utils as pu

sys.stderr.write(f"[pipeline {_time.strftime('%H:%M:%S')}] imports done\n")
sys.stderr.flush()

REPO = pu.REPO_ROOT
PY = sys.executable

STEPS: list[tuple[str, str]] = [
    ("00", "00_crawl_mindat.py"),
    ("01", "01_stac_search_download.py"),
    ("02", "02_enmap_process.py"),
    ("03", "03_clip_mines.py"),
    ("04", "04_osm_mask_batch.py"),
    ("05", "05_build_dataset.py"),
]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _tee_run(cmd: Sequence[str], log_fp: TextIO, label: str) -> int:
    """
    Run cmd unbuffered, stream its stdout/stderr to console AND log_fp,
    return the child's exit code.
    """
    line = "$ " + " ".join(shlex.quote(c) for c in cmd)
    header = f"\n[{_ts()}] >>> step {label} START\n{line}\n"
    print(header, end="", flush=True)
    log_fp.write(header)
    log_fp.flush()

    t0 = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fp.write(line)
        log_fp.flush()
    rc = proc.wait()

    elapsed = time.monotonic() - t0
    footer = (
        f"[{_ts()}] <<< step {label} END  exit={rc}  elapsed={_hms(elapsed)}\n"
    )
    print(footer, end="", flush=True)
    log_fp.write(footer)
    log_fp.flush()
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full EnMAP mineral-classification pipeline.")
    ap.add_argument("--mineral", required=True)
    # crawl knobs
    ap.add_argument("--max-pages", type=int, default=50, help="(unused; kept for compatibility)")
    ap.add_argument("--headed", action="store_true", help="Step 00: show the browser (to pass Cloudflare).")
    ap.add_argument("--xvfb", action="store_true", help="Step 00: headed Chrome in a virtual display (Linux server).")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Step 01: stop once N localities have a downloaded scene.")
    ap.add_argument("--max-localities", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.5)
    # stac knobs
    ap.add_argument("--start", default=None, help="STAC start date YYYY-MM-DD.")
    ap.add_argument("--end", default=None, help="STAC end date YYYY-MM-DD.")
    ap.add_argument("--cloud-max", type=float, default=10.0)
    # processor knobs
    ap.add_argument("--keep-only-swir", action="store_true")
    ap.add_argument("--version", default="v1", choices=["v1", "128", "58"])
    ap.add_argument("--apply-cloud-mask", action="store_true")
    ap.add_argument("--load-quality-layers", action="store_true")
    # clip knobs
    ap.add_argument("--buffer-m", type=float, default=1500.0)
    ap.add_argument("--chip-px", type=int, default=None)
    # mask knobs
    ap.add_argument("--all-touched", action="store_true")
    # dataset knobs
    ap.add_argument("--chip-size", type=int, default=128, help="Chip size enforced by step 05.")
    ap.add_argument(
        "--any-size",
        action="store_true",
        help="Step 05 accepts any chip size (skip the filter).",
    )
    ap.add_argument("--overwrite-dataset", action="store_true", help="Overwrite existing .npz files.")
    # general
    ap.add_argument(
        "--skip-step",
        action="append",
        default=[],
        choices=[s[0] for s in STEPS],
        help="Skip a step by its number (can be passed multiple times).",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="Override log file path (default: data/<mineral>/pipeline.log).",
    )
    args = ap.parse_args()

    skip = set(args.skip_step)

    def script(name: str) -> Path:
        return REPO / name

    # Ensure mineral data dir exists + pick log file path
    mdir = pu.mineral_dir(args.mineral)
    log_path = Path(args.log_file) if args.log_file else (mdir / "pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # We append (don't truncate) so resumed runs accumulate history.
    pipeline_t0 = time.monotonic()
    timings: list[tuple[str, float, int]] = []
    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(
            f"\n========== pipeline start {_ts()} mineral={args.mineral} =========="
            f"\nargs={vars(args)}\n"
        )
        log_fp.flush()
        print(f"[pipeline] log -> {log_path}", flush=True)

        # 00 crawl
        if "00" not in skip:
            cmd = [PY, "-u", str(script("00_crawl_mindat.py")), "--mineral", args.mineral,
                   "--sleep", str(args.sleep), "--resume"]
            if args.headed:
                cmd += ["--headed"]
            if args.xvfb:
                cmd += ["--xvfb"]
            if args.max_localities is not None:
                cmd += ["--max-localities", str(args.max_localities)]
            t0 = time.monotonic()
            rc = _tee_run(cmd, log_fp, "00 crawl")
            timings.append(("00 crawl", time.monotonic() - t0, rc))
            if rc != 0:
                raise SystemExit(f"step 00 failed (exit {rc})")

        # 01 stac
        if "01" not in skip:
            cmd = [PY, "-u", str(script("01_stac_search_download.py")), "--mineral", args.mineral,
                   "--cloud-max", str(args.cloud_max)]
            if args.max_samples:
                cmd += ["--target", str(args.max_samples)]
            if args.start:
                cmd += ["--start", args.start]
            if args.end:
                cmd += ["--end", args.end]
            t0 = time.monotonic()
            rc = _tee_run(cmd, log_fp, "01 stac")
            timings.append(("01 stac", time.monotonic() - t0, rc))
            if rc != 0:
                raise SystemExit(f"step 01 failed (exit {rc})")

        # 02 process
        if "02" not in skip:
            cmd = [PY, "-u", str(script("02_enmap_process.py")), "--mineral", args.mineral,
                   "--version", args.version]
            if args.keep_only_swir:
                cmd += ["--keep-only-swir"]
            if args.apply_cloud_mask:
                cmd += ["--apply-cloud-mask"]
            if args.load_quality_layers:
                cmd += ["--load-quality-layers"]
            t0 = time.monotonic()
            rc = _tee_run(cmd, log_fp, "02 process")
            timings.append(("02 process", time.monotonic() - t0, rc))
            if rc != 0:
                raise SystemExit(f"step 02 failed (exit {rc})")

        # 03 clip
        if "03" not in skip:
            cmd = [PY, "-u", str(script("03_clip_mines.py")), "--mineral", args.mineral]
            if args.chip_px is not None:
                cmd += ["--chip-px", str(args.chip_px)]
            else:
                cmd += ["--buffer-m", str(args.buffer_m)]
            t0 = time.monotonic()
            rc = _tee_run(cmd, log_fp, "03 clip")
            timings.append(("03 clip", time.monotonic() - t0, rc))
            if rc != 0:
                raise SystemExit(f"step 03 failed (exit {rc})")

        # 04 mask
        if "04" not in skip:
            cmd = [PY, "-u", str(script("04_osm_mask_batch.py")), "--mineral", args.mineral]
            if args.all_touched:
                cmd += ["--all-touched"]
            t0 = time.monotonic()
            rc = _tee_run(cmd, log_fp, "04 mask")
            timings.append(("04 mask", time.monotonic() - t0, rc))
            if rc != 0:
                raise SystemExit(f"step 04 failed (exit {rc})")

        # 05 build dataset
        if "05" not in skip:
            cmd = [PY, "-u", str(script("05_build_dataset.py")), "--mineral", args.mineral]
            if args.any_size:
                cmd += ["--any-size"]
            else:
                cmd += ["--chip-size", str(args.chip_size)]
            if args.overwrite_dataset:
                cmd += ["--overwrite"]
            t0 = time.monotonic()
            rc = _tee_run(cmd, log_fp, "05 dataset")
            timings.append(("05 dataset", time.monotonic() - t0, rc))
            if rc != 0:
                raise SystemExit(f"step 05 failed (exit {rc})")

        total = time.monotonic() - pipeline_t0
        summary = "\n[pipeline] timing summary:\n"
        for name, secs, rc in timings:
            summary += f"  {name:<14} {_hms(secs):>10}   exit={rc}\n"
        summary += f"  {'TOTAL':<14} {_hms(total):>10}\n"
        summary += (
            f"[pipeline] DONE for {args.mineral} -> {pu.mineral_dir(args.mineral)}\n"
            f"[pipeline] log file: {log_path}\n"
        )
        print(summary, end="", flush=True)
        log_fp.write(summary)


if __name__ == "__main__":
    main()
