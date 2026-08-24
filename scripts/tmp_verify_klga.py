"""Verify series 10005 is KLGA, and scan public-search for ANY pre-2025 weather markets."""
import json
import time
import requests

GAMMA = "https://gamma-api.polymarket.com"

def get_json(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)

# 1) Check an event's series + markets to confirm station (KLGA/LaGuardia?)
r = get_json(f"{GAMMA}/events", {"slug": "highest-temperature-in-nyc-on-jan-22"})
ev = r[0] if isinstance(r, list) and r else {}
print("event id:", ev.get("id"))
print("event slug:", ev.get("slug"))
print("title:", ev.get("title"))
print("seriesSlug:", ev.get("seriesSlug"))
print("series:", ev.get("series"))
print("description (first 300):")
print((ev.get("description") or "")[:300])
ms = ev.get("markets") or []
print(f"markets: {len(ms)}")
if ms:
    m0 = ms[0]
    print("first market slug:", m0.get("slug"))
    print("first market question:", (m0.get("question") or "")[:150])
    print("resolution source:", ev.get("resolutionSource"))
print("=" * 70)

# 2) Broad scan: public-search variants, collect ALL events, look for pre-2025 creation
queries = [
    "highest temperature in nyc",
    "high temperature nyc",
    "will the high temperature in nyc",
    "nyc high temp",
    "new york city high temperature",
    "temperature in nyc",
]
seen = {}
for q in queries:
    payload = get_json(f"{GAMMA}/public-search", {
        "q": q, "events_status": "closed", "keep_closed_markets": 1,
        "limit_per_type": 100,
    })
    events = payload.get("events", []) if isinstance(payload, dict) else []
    for e in events:
        cid = e.get("id")
        if cid and cid not in seen:
            seen[cid] = e
    print(f"query '{q}': {len(events)} events (cumulative unique: {len(seen)})")
    time.sleep(0.3)

# classify by creation year
from collections import Counter
years = Counter()
pre2025 = []
for cid, e in seen.items():
    cd = e.get("creationDate") or ""
    y = cd[:4]
    years[y] += 1
    if y and int(y) < 2025:
        pre2025.append(e)

print("\ncreationDate year distribution across all search hits:")
print(dict(sorted(years.items())))
print(f"\nPRE-2025 events found: {len(pre2025)}")
for e in sorted(pre2025, key=lambda x: x.get("creationDate") or "")[:20]:
    print(f"  id={e.get('id')}  created={e.get('creationDate')}  slug={e.get('slug')}")
