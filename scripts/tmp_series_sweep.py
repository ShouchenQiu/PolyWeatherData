"""Probe series IDs near 10005/10006 to find all daily-weather series created
in the same batch (2025-01-21, NYC series date). Also probe nearby IDs.

Method: fetch /series by id range 10000-10100, and by slug patterns for
many cities. Also list ALL series with 'daily' or 'weather' in slug.
"""
import sys
import time

sys.path.insert(0, r"d:\NN\PolyWeatherData")
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
            time.sleep(1.5 ** i)

print("=== 1. Series by id range 9990-10100 ===")
for sid in range(9990, 10100):
    try:
        r = get_json(f"{GAMMA}/series", {"id": sid, "limit": 1})
        if isinstance(r, list) and r:
            s = r[0]
            title = s.get("title") or ""
            slug = s.get("slug") or ""
            created = s.get("createdAt") or ""
            # only show weather/temperature related
            if any(k in (title + slug).lower() for k in ["weather", "temperature", "temp", "daily", "high"]):
                print(f"  id={sid}  {slug:35s}  created={created}  title={title[:50]}")
    except Exception:
        pass
    time.sleep(0.1)

print("\n=== 2. Series slug probes (many cities) ===")
city_slugs = [
    "nyc-daily-weather", "london-daily-weather",
    "dallas-daily-weather", "chicago-daily-weather", "seattle-daily-weather",
    "atlanta-daily-weather", "tokyo-daily-weather",
    "la-daily-weather", "los-angeles-daily-weather",
    "miami-daily-weather", "boston-daily-weather", "denver-daily-weather",
    "houston-daily-weather", "phoenix-daily-weather",
    "san-francisco-daily-weather", "sf-daily-weather",
    "paris-daily-weather", "berlin-daily-weather", "madrid-daily-weather",
    "rome-daily-weather", "amsterdam-daily-weather", "dublin-daily-weather",
    "toronto-daily-weather", "vancouver-daily-weather", "montreal-daily-weather",
    "sydney-daily-weather", "melbourne-daily-weather", "auckland-daily-weather",
    "singapore-daily-weather", "hong-kong-daily-weather", "seoul-daily-weather",
    "bangkok-daily-weather", "mumbai-daily-weather", "delhi-daily-weather",
    "mexico-city-daily-weather", "sao-paulo-daily-weather", "buenos-aires-daily-weather",
    "new-delhi-daily-weather", "washington-dc-daily-weather", "dc-daily-weather",
    "barcelona-daily-weather", "milan-daily-weather", "munich-daily-weather",
    "frankfurt-daily-weather", "zurich-daily-weather", "stockholm-daily-weather",
    "oslo-daily-weather", "helsinki-daily-weather", "warsaw-daily-weather",
    "vienna-daily-weather", "brussels-daily-weather", "lisbon-daily-weather",
    "athens-daily-weather", "istanbul-daily-weather", "dubai-daily-weather",
    "losangeles-daily-weather",
]
for slug in city_slugs:
    try:
        r = get_json(f"{GAMMA}/series", {"slug": slug, "limit": 1})
        if isinstance(r, list) and r:
            s = r[0]
            print(f"  FOUND {slug}: id={s.get('id')} created={s.get('createdAt')}")
        else:
            # try events?series=<slug>
            r2 = get_json(f"{GAMMA}/events", {"series": slug, "closed": "true", "limit": 1})
            if isinstance(r2, list) and r2:
                s2 = r2[0]
                print(f"  FOUND(events) {slug}: id={s2.get('id')} created={s2.get('creationDate')}")
    except Exception as e:
        print(f"  {slug}: ERR {type(e).__name__}")
    time.sleep(0.15)
