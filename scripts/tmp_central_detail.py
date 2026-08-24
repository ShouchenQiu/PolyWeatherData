"""Check Central Park weather markets: detail of 2021 events + probe 2022-2024 slugs."""
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

# 1) Detail of the 2021 Central Park events
for eid in ["3717", "5274"]:
    r = get_json(f"{GAMMA}/events", {"id": eid})
    ev = r[0] if isinstance(r, list) and r else {}
    print("=" * 70)
    print("event id:", eid)
    print("slug:", ev.get("slug"))
    print("title:", ev.get("title"))
    print("created:", ev.get("creationDate"))
    print("desc:", (ev.get("description") or "")[:200])
    print("resolutionSource:", ev.get("resolutionSource"))
    ms = ev.get("markets") or []
    print("markets:", len(ms))
    for m in ms[:3]:
        print("   market slug:", m.get("slug"))
        print("   question:", (m.get("question") or "")[:120])
        print("   clobTokenIds:", (m.get("clobTokenIds") or "")[:60])
    print("series:", (ev.get("series") or [{}])[0].get("slug") if ev.get("series") else None)
    time.sleep(0.3)

# 2) Probe 2022-2024 Central Park slugs (binary format)
slugs = []
for year in [2022, 2023, 2024]:
    for day in [1, 2, 3]:
        slugs.append(f"will-the-high-temperature-in-new-yorks-central-park-be-60f-or-higher-on-january-{day}-{year}")
        slugs.append(f"will-the-high-temperature-in-new-yorks-central-park-be-60f-or-higher-on-january-{day}th-{year}")

print("\n" + "=" * 70)
print("Probing 2022-2024 Central Park slugs:")
for s in slugs:
    r = get_json(f"{GAMMA}/events", {"slug": s})
    n = len(r) if isinstance(r, list) else 0
    if n:
        print(f"  FOUND: {s} -> {len(r)} events")
        for e in r[:1]:
            print(f"     id={e.get('id')} created={e.get('creationDate')} title={e.get('title')}")
    time.sleep(0.15)
print("done probing")
