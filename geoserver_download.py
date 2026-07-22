#!/usr/bin/env python3
"""
EnMAP download via DLR GeoServer STAC API (OGC API - Features / STAC).

Vendored from the user's reference implementation.

Key behavior:
- STAC search is performed without authentication.
- Asset downloads may redirect to DLR EOC SSO (sso.eoc.dlr.de). If ENMAP_USERNAME
  and ENMAP_PASSWORD are set in the environment, an SSO form login is attempted
  and cookies are reused for subsequent downloads.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import json
import logging
import os
import shutil
import sys
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx

# Load .env so ENMAP_USERNAME / ENMAP_PASSWORD are available (optional)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


STAC_BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION_ENMAP_L2A = "ENMAP_HSI_L2A"
SEARCH_URL = f"{STAC_BASE}/search"

OVERALL_QUALITY_LABELS = {0: "nominal", 1: "reduced", 2: "low"}


def _get_overall_quality(item: Dict[str, Any]) -> Optional[int]:
    props = item.get("properties") or {}
    q = props.get("enmap:overallQuality", props.get("enmap:OverallQuality"))
    return int(q) if q is not None else None


def get_logger(name: str = "geoserver_download", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def search_stac(
    geometry: Dict[str, Any],
    *,
    collection: str = COLLECTION_ENMAP_L2A,
    datetime_range: Optional[str] = None,
    limit: int = 10,
    max_cloud_cover: Optional[float] = None,
    max_overall_quality: Optional[int] = None,
    sortby_cloud_asc: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    log = logger or get_logger()
    body: Dict[str, Any] = {
        "collections": [collection],
        "intersects": geometry,
        "limit": min(limit, 10000),
    }
    if datetime_range:
        body["datetime"] = datetime_range

    query_parts: Dict[str, Any] = {}
    if max_cloud_cover is not None:
        query_parts["eo:cloud_cover"] = {"lte": float(max_cloud_cover)}
        log.info("Cloud cover filter: <= %.1f%%", max_cloud_cover)
    if max_overall_quality is not None:
        query_parts["enmap:overallQuality"] = {"lte": int(max_overall_quality)}
        log.info(
            "Overall quality filter: enmap:overallQuality <= %s (0=nominal, 1=reduced, 2=low)",
            max_overall_quality,
        )
    if query_parts:
        body["query"] = query_parts

    if sortby_cloud_asc:
        body["sortby"] = [{"field": "properties.eo:cloud_cover", "direction": "asc"}]

    log.info(
        "STAC search: collection=%s datetime=%s limit=%s max_cloud_cover=%s max_overall_quality=%s",
        collection,
        datetime_range,
        limit,
        max_cloud_cover,
        max_overall_quality,
    )
    with httpx.Client(timeout=60) as client:
        r = client.post(
            SEARCH_URL,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/geo+json"},
        )
        r.raise_for_status()
        data = r.json()
    features = data.get("features") or []
    log.info("STAC returned %d items", len(features))
    return features


_QUICKLOOK_INDICATORS = frozenset(
    {"quicklook", "thumbnail", "visual", "preview", "overview", "browse", "thumb"}
)
_USERNAME_NAMES = ("username", "user", "email", "j_username", "login", "loginid", "uid")
_PASSWORD_NAMES = ("password", "passwd", "j_password", "pwd", "pass")
_SSO_FALLBACK_URLS = (
    "https://sso.eoc.dlr.de/cas/login",
    "https://sso.eoc.dlr.de/login",
    "https://sso.eoc.dlr.de/",
)


def _is_low_quality_asset(key: str, asset: Dict[str, Any]) -> bool:
    key_lower = key.lower()
    if any(x in key_lower for x in _QUICKLOOK_INDICATORS):
        return True
    roles = (asset.get("roles") or [])
    if "thumbnail" in roles or "overview" in roles:
        return True
    title = (asset.get("title") or "").lower()
    if any(x in title for x in ("quicklook", "thumbnail", "preview", "overview")):
        return True
    return False


def _content_looks_like_html(content: bytes) -> bool:
    return bool(content.startswith(b"<!") or content.startswith(b"<html") or content.startswith(b"<!DOCTYPE"))


def _response_is_html(response: httpx.Response, content: bytes) -> bool:
    ct = (response.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        return True
    return _content_looks_like_html(content)


class _LoginFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.form_action: Optional[str] = None
        self.inputs: Dict[str, str] = {}
        self._username_name: Optional[str] = None
        self._password_name: Optional[str] = None
        self._in_form = False
        self._current_form_action: Optional[str] = None
        self._current_form_inputs: Dict[str, str] = {}
        self._current_username: Optional[str] = None
        self._current_password: Optional[str] = None
        self._found_form = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        ad = dict(attrs)
        if tag == "form":
            self._in_form = True
            self._current_form_action = ad.get("action") or ""
            self._current_form_inputs = {}
            self._current_username = None
            self._current_password = None
        if tag == "input" and self._in_form:
            name = ad.get("name")
            if name:
                self._current_form_inputs[name] = ad.get("value") or ""
                kind = (ad.get("type") or "text").lower()
                name_lower = name.lower()
                if kind == "password" or name_lower in _PASSWORD_NAMES:
                    self._current_password = name
                elif (kind in ("text", "email", "") or name_lower in _USERNAME_NAMES) and self._current_username is None:
                    self._current_username = name

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._in_form:
            self._in_form = False
            if (
                not self._found_form
                and self._current_form_action is not None
                and self._current_username
                and self._current_password
            ):
                self._found_form = True
                self.form_action = self._current_form_action
                self.inputs = dict(self._current_form_inputs)
                self._username_name = self._current_username
                self._password_name = self._current_password


def _debug_save_login_html(html: str, path: Optional[Path] = None) -> None:
    if not os.environ.get("ENMAP_DEBUG"):
        return
    p = path or Path("login_page_debug.html")
    try:
        p.write_text(html, encoding="utf-8")
        print(f"[ENMAP_DEBUG] Wrote login page to {p}", file=sys.stderr)
    except Exception:
        pass


def _do_sso_login_with_client(
    client: httpx.Client,
    login_url: str,
    username: str,
    password: str,
    logger: logging.Logger,
) -> None:
    r = client.get(login_url)
    r.raise_for_status()
    _debug_save_login_html(r.text)
    parser = _LoginFormParser()
    parser.feed(r.text)
    if not parser.form_action or not parser._username_name or not parser._password_name:
        raise RuntimeError(
            "Could not find login form (missing action or username/password fields). "
            "Set ENMAP_DEBUG=1 and inspect login_page_debug.html."
        )
    post_url = urljoin(login_url, parser.form_action)
    data = dict(parser.inputs)
    data[parser._username_name] = username
    data[parser._password_name] = password
    if os.environ.get("ENMAP_DEBUG"):
        logger.info("Login form: post_url=%s, username_field=%s", post_url, parser._username_name)
    r2 = client.post(post_url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    r2.raise_for_status()


def _get_login_url_from_response(r: httpx.Response) -> Optional[str]:
    if r.status_code == 302:
        loc = (r.headers.get("location") or "").strip()
        if "sso.eoc.dlr.de" in loc or "login" in loc.lower():
            return loc
    if r.status_code == 200 and _response_is_html(r, r.content):
        return str(r.url)
    return None


def _authenticate_client(
    client: httpx.Client,
    href: str,
    username: str,
    password: str,
    logger: logging.Logger,
) -> bool:
    r0 = client.get(href, follow_redirects=False)
    login_url = _get_login_url_from_response(r0)
    if not login_url:
        return False
    urls_to_try: List[str] = [login_url]
    if "sso.eoc.dlr.de" not in login_url:
        urls_to_try.extend(_SSO_FALLBACK_URLS)
    last_error: Optional[Exception] = None
    for url in urls_to_try:
        try:
            _do_sso_login_with_client(client, url, username, password, logger)
            return True
        except RuntimeError as e:
            last_error = e
            if "Could not find login form" in str(e):
                continue
            raise
    if last_error:
        raise last_error
    return False


def _get_credentials(username: Optional[str] = None, password: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    u = username if username is not None else os.environ.get("ENMAP_USERNAME")
    p = password if password is not None else os.environ.get("ENMAP_PASSWORD")
    if u is not None:
        u = u.strip() or None
    if p is not None:
        p = p.strip() or None
    return (u, p)


def _get_all_downloadable_assets(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    assets = item.get("assets") or {}
    out: List[Tuple[str, str]] = []
    for key, asset in assets.items():
        if not isinstance(asset, dict):
            continue
        href = asset.get("href")
        if not href:
            continue
        if _is_low_quality_asset(key, asset):
            continue
        out.append((key, href))
    return out


ENMAP_FILENAME_RENAME: Dict[str, str] = {
    "metadata.xml": "METADATA.XML",
    "image.tif": "SPECTRAL_IMAGE.TIF",
    "quality_cloud_shadow.tif": "QL_QUALITY_CLOUDSHADOW.TIF",
    "quality_classes.tif": "QL_QUALITY_CLASSES.TIF",
    "quality_cloud.tif": "QL_QUALITY_CLOUD.TIF",
    "quality_haze.tif": "QL_QUALITY_HAZE.TIF",
    "quality_cirrus.tif": "QL_QUALITY_CIRRUS.TIF",
    "quality_snow.tif": "QL_QUALITY_SNOW.TIF",
    "quality_testflags.tif": "QL_QUALITY_TESTFLAGS.TIF",
    "defective_pixel_mask.tif": "QL_PIXELMASK.TIF",
}


def _zip_member_name(asset_key: str, href: str) -> str:
    base = href.split("/")[-1].split("?")[0]
    if "." in base:
        ext = base.rsplit(".", 1)[-1]
        raw_name = f"{asset_key}.{ext}"
    else:
        raw_name = asset_key
    key_lower = raw_name.lower()
    return ENMAP_FILENAME_RENAME.get(key_lower, raw_name)


def _fetch_asset_content(
    href: str,
    *,
    session: Optional[httpx.Client],
    username: Optional[str],
    password: Optional[str],
    logger: logging.Logger,
) -> Optional[bytes]:
    if session is not None:
        r = session.get(href)
        r.raise_for_status()
        return None if _response_is_html(r, r.content) else r.content

    creds_user, creds_pass = _get_credentials(username, password)
    if creds_user and creds_pass:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            try:
                if _authenticate_client(client, href, creds_user, creds_pass, logger):
                    r = client.get(href)
                    r.raise_for_status()
                    if not _response_is_html(r, r.content):
                        return r.content
            except RuntimeError:
                pass

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        r = client.get(href)
        r.raise_for_status()
        return None if _response_is_html(r, r.content) else r.content


def download_item(
    item: Dict[str, Any],
    output_dir: Path,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    session: Optional[httpx.Client] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    log = logger or get_logger()
    item_id = item.get("id") or "unknown"
    all_assets = _get_all_downloadable_assets(item)
    if not all_assets:
        log.warning("No downloadable assets for item %s", item_id)
        return False

    session_to_use = session
    own_session: Optional[httpx.Client] = None
    if session_to_use is None:
        creds_user, creds_pass = _get_credentials(username, password)
        if creds_user and creds_pass:
            first_href = all_assets[0][1]
            own_session = httpx.Client(timeout=60, follow_redirects=True)
            try:
                if _authenticate_client(own_session, first_href, creds_user, creds_pass, log):
                    session_to_use = own_session
            except RuntimeError as e:
                log.warning("Authentication failed: %s", e)
                own_session.close()
                own_session = None

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{item_id}.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for asset_key, href in all_assets:
                content = _fetch_asset_content(
                    href,
                    session=session_to_use,
                    username=username,
                    password=password,
                    logger=log,
                )
                if not content:
                    continue
                member_name = _zip_member_name(asset_key, href)
                zf.writestr(member_name, content)
    finally:
        if own_session is not None:
            own_session.close()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    if not names:
        zip_path.unlink(missing_ok=True)
        log.error(
            "Server returned HTML for all assets (likely login required). "
            "Check ENMAP_USERNAME and ENMAP_PASSWORD in .env. Skipping %s",
            item_id,
        )
        return False

    q = _get_overall_quality(item)
    if q is not None:
        label = OVERALL_QUALITY_LABELS.get(q, str(q))
        log.info("Saved: %s (%d files) | enmap:overallQuality=%s (%s)", zip_path.name, len(names), q, label)
    else:
        log.info("Saved: %s (%d files)", zip_path.name, len(names))
    return True


def download_enmap_from_points(
    latlon_dict: Dict[str, Tuple[float, float]],
    output_root: Path,
    *,
    buffer_m: float = 500,
    datetime_range: Optional[str] = None,
    limit: Optional[int] = None,
    max_cloud_cover: Optional[float] = None,
    max_overall_quality: Optional[int] = None,
    sortby_cloud_asc: bool = True,
    remove_empty_point_dirs: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Optional[Path]]:
    log = logger or get_logger()
    if not latlon_dict:
        raise ValueError("latlon_dict is empty")
    if buffer_m <= 0:
        raise ValueError("buffer_m must be > 0")

    # We intentionally avoid geopandas/shapely in this vendored copy. The calling
    # pipeline already knows how to build a Point geometry dict for STAC intersects,
    # so this helper remains only for compatibility with the original script API.
    raise RuntimeError(
        "download_enmap_from_points is not supported in the vendored module. "
        "Use search_stac() + download_item() directly."
    )

