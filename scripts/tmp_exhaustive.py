"""Exhaustive probe for ANY NYC/New York weather markets in 2022-2024.

Try many slug variants: different stations (central-park, nyc, new-york, la-guardia,
jfk, laguardia), different thresholds (40/50/60/70f), different formats.
Also public-search with year keywords.
"""
import json
import time
import requests

GAMMA = "https://gamma-api.polymarket.com"

def get_json(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 ** i)

found = []

# Variant slugs for 2022/2023/2024
stations = ["new-yorks-central-park", "nyc", "new-york-city", "new-york", "ny", "la-guardia", "laguardia", "jfk"]
thresh = [40, 50, 60, 70, 80, 90, 100]
years = [2022, 2023, 2024]
days = [1, 15]

slug_variants = []
for y in years:
    for station in stations:
        for t in thresh:
            for d in days:
                slug_variants.append(f"will-the-high-temperature-in-{station}-be-{t}f-or-higher-on-january-{d}-{y}")
                slug_variants.append(f"will-the-high-temperature-in-{station}-be-{t}f-or-higher-on-january-{d}th-{y}")
                slug_variants.append(f"will-the-high-temperature-in-{station}-be-{t}f-or-higher-on-july-{d}-{y}")

print(f"probing {len(slug_variants)} slug variants...")
hit = 0
for s in slug_variants:
    try:
        r = get_json(f"{GAMMA}/events", {"slug": s})
    except Exception:
        continue
    if isinstance(r, list) and r:
        hit += 1
        found.append(r[0])
        print(f"  FOUND: {s} (id={r[0].get('id')} created={r[0].get('creationDate')})")
    time.sleep(0.05)

print(f"\ntotal slug hits: {hit}")

# public-search with year keywords
print("\npublic-search with year keywords:")
for q in ["high temperature nyc 2023", "high temperature nyc 2022", "high temperature new york 2023",
          "nyc weather 2023 temperature", "temperature nyc december 2023"]:
    try:
        payload = get_json(f"{GAMMA}/public-search", {
            "q": q, "events_status": "closed", "keep_closed_markets": 1, "limit_per_type": 100,
        })
        events = payload.get("events", []) if isinstance(payload, dict) else []
        for e in events:
            slug = str(e.get("slug") or "")
            title = str(e.get("title") or "")
            if "temperature" in slug or "high-temp" in slug or "temperature" in title.lower():
                if e.get("id") not in [f.get("id") for f in found]:
                    found.append(e)
        print(f"  '{q}': {len(events)} events")
        time.sleep(0.3)
    except Exception as ex:
        print(f"  '{q}': ERR {ex}")

print("\nALL found pre-KLGA-series weather events:")
for e in sorted(found, key=lambda x: x.get("creationDate") or "9999"):
    print(f"  id={e.get('id')}  created={e.get('creationDate')}  {e.get('slug')}")
