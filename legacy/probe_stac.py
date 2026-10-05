#!/usr/bin/env python3
"""Find a request shape the DLR STAC server accepts for bulk paging."""
import json
import httpx

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLL = "ENMAP_HSI_L2A"
BBOX = [20.0, 40.0, 30.0, 50.0]  # Greece / Balkans, certain to have scenes
POLY = {"type": "Polygon", "coordinates": [[[20.0, 40.0], [30.0, 40.0], [30.0, 50.0], [20.0, 50.0], [20.0, 40.0]]]}
DT = "2022-04-01T00:00:00Z/2026-09-28T23:59:59Z"

CASES = [
    ("POST bbox limit=100",            {"collections": [COLL], "bbox": BBOX, "limit": 100}),
    ("POST bbox limit=500",            {"collections": [COLL], "bbox": BBOX, "limit": 500}),
    ("POST bbox+datetime limit=100",   {"collections": [COLL], "bbox": BBOX, "datetime": DT, "limit": 100}),
    ("POST intersects limit=100",      {"collections": [COLL], "intersects": POLY, "limit": 100}),
    ("POST intersects limit=500",      {"collections": [COLL], "intersects": POLY, "limit": 500}),
    ("POST intersects+datetime l=100", {"collections": [COLL], "intersects": POLY, "datetime": DT, "limit": 100}),
    ("POST intersects+dt limit=1000",  {"collections": [COLL], "intersects": POLY, "datetime": DT, "limit": 1000}),
]

with httpx.Client(timeout=120) as c:
    for name, body in CASES:
        try:
            r = c.post(f"{BASE}/search", json=body,
                       headers={"Content-Type": "application/json", "Accept": "application/geo+json"})
            if r.status_code == 200:
                d = r.json()
                feats = d.get("features") or []
                links = [l.get("rel") for l in (d.get("links") or [])]
                nxt = next((l for l in (d.get("links") or []) if l.get("rel") == "next"), None)
                shape = "href" if (nxt or {}).get("href") else ("body" if (nxt or {}).get("body") else "none")
                print(f"OK   {name}: {len(feats)} feats, matched={d.get('numberMatched')}, "
                      f"links={links}, next={shape}")
                if nxt:
                    print(f"       next -> {json.dumps(nxt)[:300]}")
            else:
                print(f"{r.status_code}  {name}: {r.text[:200]}")
        except Exception as e:
            print(f"ERR  {name}: {e!r}")

    # GET variants
    for name, url in [
        ("GET /search bbox", f"{BASE}/search?collections={COLL}&bbox={','.join(map(str, BBOX))}&limit=100"),
        ("GET items bbox", f"{BASE}/collections/{COLL}/items?bbox={','.join(map(str, BBOX))}&limit=100"),
    ]:
        try:
            r = c.get(url, headers={"Accept": "application/geo+json"})
            if r.status_code == 200:
                d = r.json()
                nxt = next((l for l in (d.get("links") or []) if l.get("rel") == "next"), None)
                print(f"OK   {name}: {len(d.get('features') or [])} feats, "
                      f"matched={d.get('numberMatched')}, next={(nxt or {}).get('href', 'none')[:120]}")
            else:
                print(f"{r.status_code}  {name}: {r.text[:200]}")
        except Exception as e:
            print(f"ERR  {name}: {e!r}")
