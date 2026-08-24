"""Dig deeper: Central Park / binary-format NYC weather markets in 2021-2024."""
import json
import time
import requests
from collections import Counter

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

queries = [
    "high temperature new yorks central park",
    "central park temperature",
    "will the high temperature",
    "high temperature be or higher",
    "temperature in new yorks central park",
    "be 60f or higher",
    "high-temp new york",
]
seen = {}
for q in queries:
    try:
        payload = get_json(f"{GAMMA}/public-search", {
            "q": q, "events_status": "closed", "keep_closed_markets": 1,
            "limit_per_type": 100,
        })
        events = payload.get("events", []) if isinstance(payload, dict) else []
    except Exception as e:
        print(f"query '{q}': ERROR {e}")
        continue
    new = 0
    for e in events:
        cid = e.get("id")
        if cid and cid not in seen:
            seen[cid] = e
            new += 1
    print(f"query '{q}': {len(events)} events (+{new} new, total {len(seen)})")
    time.sleep(0.3)

# Filter to weather-temperature-ish events
weather = []
for cid, e in seen.items():
    slug = str(e.get("slug") or "")
    title = str(e.get("title") or "")
    if any(k in slug for k in ["temperature", "high-temp", "snow", "rain", "storm", "weather"]):
        weather.append(e)
    elif any(k in title for k in ["temperature", "high temperature", "snow", "weather"]):
        weather.append(e)

print(f"\ntemperature/weather-related unique events: {len(weather)}")
years = Counter()
for e in weather:
    y = (e.get("creationDate") or "")[:4]
    years[y] += 1
print("by creation year:", dict(sorted(years.items())))

print("\nAll temperature-related events sorted by creationDate:")
for e in sorted(weather, key=lambda x: x.get("creationDate") or "9999"):
    print(f"  id={e.get('id'):>8}  created={e.get('creationDate')}  {e.get('slug')}")
