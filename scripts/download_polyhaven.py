"""Download tonemapped equirectangular JPGs of Poly Haven HDRIs (CC0) into data/pano/.

python scripts/download_polyhaven.py --n 80 --category indoor
Uses the public Poly Haven API (https://api.polyhaven.com). Check the 'tonemapped' key
if the API format changes.
"""
import argparse
import json
import os
import random
import urllib.request

API = "https://api.polyhaven.com"
UA = {"User-Agent": "sync-flowedit-course-project"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--category", default="", help="e.g. indoor, outdoor, urban, nature")
    ap.add_argument("--out", default="data/pano")
    ap.add_argument("--seed", type=int, default=0, help="random subset (reproducible)")
    ap.add_argument("--exclude", default="", help="skip assets that also have this category, e.g. indoor")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q = f"{API}/assets?t=hdris" + (f"&c={a.category}" if a.category else "")
    meta = json.loads(get(q))
    ids = sorted(k for k, v in meta.items() if not (a.exclude and a.exclude in v.get("categories", [])))
    random.Random(a.seed).shuffle(ids)
    assets = ids[: a.n]
    with open(os.path.join(a.out, f"list_{a.category or 'all'}.txt"), "w") as f:
        f.write("\n".join(assets) + "\n")
    for aid in assets:
        dst = os.path.join(a.out, f"{aid}.jpg")
        if os.path.exists(dst):
            continue
        files = json.loads(get(f"{API}/files/{aid}"))
        url = files.get("tonemapped", {}).get("url")
        if not url:
            print("skip (no tonemapped jpg):", aid)
            continue
        open(dst, "wb").write(get(url))
        print("ok", aid)


if __name__ == "__main__":
    main()
