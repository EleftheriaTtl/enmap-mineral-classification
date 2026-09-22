#!/usr/bin/env python3
"""
00_crawl_mindat.py
==================

Step 00: crawl Mindat for mine localities of a mineral.

This implementation is adapted from the user's Archive `mindat_scrape.py`:
- SeleniumBase (undetected-chromedriver)
- optional cookie persistence
- Cloudflare challenge awareness (headed run required to solve)
- incremental atomic CSV saving

Output (pipeline-compatible):
  data/<mineral>/localities.csv

Columns:
  Name, lat, lon, Type, MineralsStr, Description, References, LocalityURL,
  Timestamp, StartedAt, FinishedAt, DurationSec
"""

from __future__ import annotations

import argparse
import pickle
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

import pipeline_utils as pu

try:
    from seleniumbase import SB
    from selenium.webdriver.common.keys import Keys
except Exception as e:  # pragma: no cover
    SB = None  # type: ignore
    Keys = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", file=sys.stderr, flush=True)


def save_df_atomic(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)


def challenge_present(sb) -> bool:
    try:
        if "just a moment" in (sb.get_title() or "").lower():
            return True
        if sb.is_text_visible("Verify you are human"):
            return True
        if sb.is_text_visible("Large number of pages viewed"):
            return True
    except Exception:
        pass
    for sel in (
        'iframe[src*="challenges"]',
        'iframe[title*="Cloudflare"]',
        "input[value*='Verify']",
        "button:contains('Verify you are human')",
    ):
        try:
            if sb.is_element_visible(sel):
                return True
        except Exception:
            pass
    return False


XVFB = False  # set by --xvfb: headed Chrome in a virtual display (Linux servers)


def wait_until_home_ready(sb, *, headed: bool, overall_timeout: int = 180) -> None:
    start = time.time()
    sb.uc_open_with_reconnect("https://www.mindat.org/", 3)

    while True:
        try:
            sb.wait_for_element_visible("#searchboxhp", timeout=2)
            return
        except Exception:
            pass

        if challenge_present(sb) and XVFB:
            _log("Cloudflare challenge detected -> clicking it (xvfb) ...")
            try:
                sb.uc_gui_click_captcha()
            except Exception as e:
                _log(f"  captcha click failed: {e!r}")
            time.sleep(3)
            continue

        if challenge_present(sb):
            if not headed:
                raise RuntimeError(
                    "Cloudflare challenge detected in headless mode. "
                    "Run once with --headed to solve it, then reuse cookies."
                )
            _log("Cloudflare challenge detected — please solve it in the browser window.")
            while time.time() - start < overall_timeout:
                try:
                    sb.wait_for_element_visible("#searchboxhp", timeout=2)
                    _log("Challenge cleared. Continuing.")
                    return
                except Exception:
                    time.sleep(2)
            raise TimeoutError("Timed out waiting for challenge to be solved.")

        if time.time() - start > overall_timeout:
            raise TimeoutError("Timed out waiting for the Mindat home search box.")
        time.sleep(2)
        sb.refresh()
        time.sleep(1)


def open_and_search(sb, mineral: str, *, headed: bool, pause: float) -> None:
    wait_until_home_ready(sb, headed=headed, overall_timeout=180)
    sb.clear("#searchboxhp")
    sb.type("#searchboxhp", mineral + Keys.RETURN)
    time.sleep(pause)
    if challenge_present(sb):
        if not headed:
            raise RuntimeError(
                "Cloudflare challenge after search in headless mode. "
                "Run once with --headed to solve it, then reuse cookies."
            )
        _log("Challenge appeared after search — please solve it.")
        start = time.time()
        while time.time() - start < 180:
            time.sleep(2)
            if not challenge_present(sb):
                break
    time.sleep(pause)


def save_cookies(sb, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(sb.driver.get_cookies(), open(path, "wb"))


def load_cookies(sb, path: Path) -> None:
    cookies = pickle.load(open(path, "rb"))
    sb.uc_open_with_reconnect("https://www.mindat.org/", 3)
    for c in cookies:
        allowed = {
            k: c.get(k)
            for k in ("name", "value", "domain", "path", "expiry", "secure", "httpOnly", "sameSite")
        }
        try:
            sb.driver.add_cookie(allowed)
        except Exception:
            allowed.pop("sameSite", None)
            allowed.pop("expiry", None)
            sb.driver.add_cookie(allowed)
    sb.refresh()


def open_mineral_page_from_current(sb, mineral: str, *, pause: float) -> bool:
    """Find and open a /min-XXXX.html link from the current page."""
    soup = BeautifulSoup(sb.get_page_source(), "lxml")
    cand: list[tuple[str, str]] = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if not re.search(r"/min-\d+\.html$", href):
            continue
        txt = a.get_text(" ", strip=True).lower()
        cand.append((txt, urljoin(sb.get_current_url(), href)))
    if not cand:
        return False
    m = mineral.strip().lower()
    cand.sort(key=lambda t: (0 if m and m in t[0] else 1, len(t[0])))
    sb.open(cand[0][1])
    time.sleep(max(0.6, pause))
    return True


def find_locality_container_id_from_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    headers = soup.find_all("h2", class_="nogloss mindatadivh2")
    target_h2 = None
    for h2 in headers:
        txt = h2.get_text(" ", strip=True)
        if txt.startswith("Locality List"):
            target_h2 = h2
            break
    if not target_h2:
        return None
    toggle = target_h2.find("a", id=re.compile(r"^collapseb\d+$"))
    if toggle and toggle.get("id"):
        num = re.search(r"\d+", toggle["id"])
        if num:
            return f"collapse{num.group(0)}"
    sib = target_h2.find_next_sibling(
        lambda tag: tag.name in ("div", "section")
        and tag.has_attr("id")
        and re.match(r"^collapse\d+$", tag["id"])
    )
    if sib:
        return sib["id"]
    return None


def extract_locality_links_from_html(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    container_id = find_locality_container_id_from_html(html)
    container = soup.select_one(f"#{container_id}") if container_id else None
    if container is None:
        container = soup  # fallback: whole page

    out: list[dict] = []
    seen: Set[str] = set()
    for a in container.select("a[href]"):
        href = a.get("href") or ""
        if not re.search(r"/loc-\d+\.html", href):
            continue
        full = urljoin(base_url, href)
        if full in seen:
            continue
        seen.add(full)
        out.append({"text": a.get_text(" ", strip=True) or full.split("/")[-1], "href": full})
    return out


def parse_latlon(s: str) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) < 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except Exception:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (lat, lon)


def get_locality_info(sb, item: Dict[str, Any], *, pause: float, headed: bool) -> Dict[str, Any]:
    sb.open(item["href"])
    time.sleep(max(0.6, pause))
    if challenge_present(sb):
        if not headed:
            raise RuntimeError("Challenge detected on locality page in headless mode.")
        _log("Challenge detected on locality page — solve it, then continue...")
        start = time.time()
        while time.time() - start < 180:
            time.sleep(2)
            if not challenge_present(sb):
                break
    time.sleep(max(0.6, pause))

    soup = BeautifulSoup(sb.get_page_source(), "lxml")
    latlon = None
    type_ = None
    locdiv = soup.select_one("#locinfodiv")
    if locdiv:
        for row in locdiv.select(".LFtr"):
            th = row.select_one(".LFth")
            td = row.select_one(".LFtd")
            if not th or not td:
                continue
            htxt = th.get_text(strip=True)
            vtxt = td.get_text(" ", strip=True)
            if "Latitude" in htxt and "Longitude" in htxt and ("decimal" in htxt or "WGS84" in htxt):
                latlon = vtxt
            elif htxt.startswith("Type"):
                type_ = vtxt

    minerals = []
    h2_mins = soup.find("h2", string=lambda t: t and "Mineral List" in t)
    if h2_mins:
        container = soup.select_one("#newlocminlist") or h2_mins.find_next("div")
        if container:
            for a in container.select("a.lm"):
                txt = a.get_text(strip=True)
                if txt:
                    minerals.append(txt)

    references = []
    h2_refs = soup.find("h2", string=lambda t: t and "References" in t)
    if h2_refs:
        container = h2_refs.find_next("div", class_="refshowbox")
        if container:
            for a in container.select("a.referencelink"):
                href = a.get("href")
                if href:
                    references.append(urljoin(item["href"], href))

    return {
        "Name": item.get("text"),
        "Latitude & Longitude": latlon,
        "Type": type_,
        "MineralsStr": ", ".join(minerals),
        "Description": None,
        "References": references,
        "LocalityURL": item["href"],
    }


def crawl(
    *,
    mineral: str,
    out_csv: Path,
    cookies_path: Optional[Path],
    headed: bool,
    pause: float,
    max_localities: Optional[int],
    resume: bool,
) -> int:
    if SB is None:
        raise RuntimeError(
            "seleniumbase is not installed. Install dependencies first:\n"
            "  pip install seleniumbase lxml\n"
            f"Import error: {_IMPORT_ERR}"
        )

    rows: list[dict] = []
    seen_urls: Set[str] = set()
    if resume and out_csv.exists():
        existing = pd.read_csv(out_csv)
        if "LocalityURL" in existing.columns:
            seen_urls = set(existing["LocalityURL"].dropna().astype(str).tolist())
        rows = existing.to_dict("records")
        _log(f"Loaded {len(rows)} existing rows from {out_csv}.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sb_kwargs = dict(uc=True, xvfb=True) if XVFB else dict(uc=True, headed=headed, headless=(not headed))
    with SB(**sb_kwargs) as sb:
        if cookies_path and cookies_path.exists():
            _log(f"Loading cookies from {cookies_path} ...")
            load_cookies(sb, cookies_path)

        _log("Opening Mindat and searching ...")
        open_and_search(sb, mineral, headed=headed, pause=pause)

        # Ensure we land on the mineral page (/min-*.html)
        if not open_mineral_page_from_current(sb, mineral, pause=pause):
            dbg = pu.REPO_ROOT / "mindat_debug_search_results.html"
            dbg.write_text(sb.get_page_source(), encoding="utf-8", errors="ignore")
            _log(f"Wrote debug HTML -> {dbg}")
            raise RuntimeError("Could not find a mineral page link after searching.")

        html = sb.get_page_source()
        links = extract_locality_links_from_html(html, sb.get_current_url())
        if not links:
            dbg = pu.REPO_ROOT / "mindat_debug_mineral_page.html"
            dbg.write_text(html, encoding="utf-8", errors="ignore")
            _log(f"Wrote debug HTML -> {dbg}")
            raise RuntimeError("No locality links found (layout may have changed).")

        if max_localities is not None:
            links = links[:max_localities]
        _log(f"Found {len(links)} locality links to visit.")

        for i, item in enumerate(links, 1):
            href = str(item.get("href", ""))
            if href in seen_urls:
                _log(f"[{i:03}/{len(links)}] SKIP (already saved): {href}")
                continue

            started_at = datetime.now().isoformat(timespec="seconds")
            t0 = time.time()
            try:
                info = get_locality_info(sb, item, pause=pause, headed=headed)
                finished_at = datetime.now().isoformat(timespec="seconds")
                duration_sec = round(time.time() - t0, 2)

                parsed = parse_latlon(info.get("Latitude & Longitude") or "")
                if not parsed:
                    _log(f"[{i:03}/{len(links)}] no-coords {href}")
                    continue
                lat, lon = parsed

                rec = {
                    "Name": info.get("Name"),
                    "lat": lat,
                    "lon": lon,
                    "Type": info.get("Type"),
                    "MineralsStr": info.get("MineralsStr"),
                    "Description": info.get("Description"),
                    "References": info.get("References"),
                    "LocalityURL": href,
                    "Timestamp": finished_at,
                    "StartedAt": started_at,
                    "FinishedAt": finished_at,
                    "DurationSec": duration_sec,
                }
                rows.append(rec)
                seen_urls.add(href)

                df_live = pd.DataFrame(rows)
                save_df_atomic(df_live, out_csv)
                _log(f"[{i:03}/{len(links)}] OK: {rec['Name']} - {rec.get('Type')} ({duration_sec}s)")
            except Exception as e:
                duration_sec = round(time.time() - t0, 2)
                _log(f"[{i:03}/{len(links)}] ERROR on {href} after {duration_sec}s: {e}")

            time.sleep(float(pause) + random.uniform(0.4, 1.2))
            if i % 30 == 0:
                _log("Cooling down for 90s ...")
                time.sleep(90)

        if headed and cookies_path:
            _log(f"Saving cookies to {cookies_path} ...")
            save_cookies(sb, cookies_path)

    df = pd.DataFrame(rows)
    want = [
        "Name",
        "lat",
        "lon",
        "Type",
        "MineralsStr",
        "Description",
        "References",
        "LocalityURL",
        "Timestamp",
        "StartedAt",
        "FinishedAt",
        "DurationSec",
    ]
    for c in want:
        if c not in df.columns:
            df[c] = None
    df = df[want]
    save_df_atomic(df, out_csv)
    _log(f"DONE: {len(df)} localities -> {out_csv}")
    return int(len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape Mindat localities for a mineral (SeleniumBase).")
    ap.add_argument("--mineral", required=True)
    ap.add_argument("--max-localities", "--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=2.0, help="Politeness delay / page settle time (seconds).")
    ap.add_argument("--cookies", default=None, help="Cookie jar path (pickle). Loaded if exists; saved in headed mode.")
    ap.add_argument("--headed", action="store_true", help="Show browser (needed to solve challenges).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--xvfb", action="store_true", help="Linux server: headed Chrome in a virtual display + auto-click Cloudflare.")
    ap.add_argument("--out", default=None, help="Override output CSV path.")
    args = ap.parse_args()

    global XVFB
    XVFB = bool(args.xvfb)
    out_csv = Path(args.out) if args.out else pu.localities_csv(args.mineral)
    cookies_path = Path(args.cookies) if args.cookies else None
    n = crawl(
        mineral=args.mineral,
        out_csv=out_csv,
        cookies_path=cookies_path,
        headed=bool(args.headed),
        pause=float(args.sleep),
        max_localities=(int(args.max_localities) if args.max_localities else None),
        resume=bool(args.resume),
    )
    print(f"{n} localities written to {out_csv}")


if __name__ == "__main__":
    main()

