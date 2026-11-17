"""Pick a small validation subset from a finished benchmark: the edits where per-view independent
FlowEdit changed the image most but Sync-FlowEdit did not (largest CLIP-gain gap), 4 indoor + 4 outdoor,
one edit per panorama, skipping sky/lighting edits (sunset/day/night: separate sky-on-ground issue).
python scripts/make_val_subset.py --n 8 --out configs/val8.yaml
"""
import argparse
import glob
import json

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="outputs/bench/flux")
ap.add_argument("--source_clip", default="outputs/bench/flux_source_clip.json")
ap.add_argument("--config", default="configs/edits.yaml")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", default="configs/val8.yaml")
a = ap.parse_args()

base = json.load(open(a.source_clip))
R = {}
for f in glob.glob(f"{a.bench}/*.json"):
    r = json.load(open(f))
    R[(r["image"], r["edit"], r["method"])] = r
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
cand = []
for p in cfg["panoramas"]:
    for e in p["edits"]:
        if e["name"] in ("sunset", "day", "night"):
            continue
        k = (p["image"], e["name"])
        b = base[f"{k[0]}|{k[1]}"]
        gi = R[k + ("independent",)]["clip_tar"] - b
        gs = R[k + ("sync",)]["clip_tar"] - b
        cand.append((gi - gs, gi, gs, p, e))
cand.sort(key=lambda x: -x[0])
pick, per_group = [], {}
for c in cand:
    g = c[3]["group"]
    if per_group.get(g, 0) < a.n // 2 and all(c[3]["image"] != q[3]["image"] for q in pick):
        pick.append(c)
        per_group[g] = per_group.get(g, 0) + 1
    if len(pick) == a.n:
        break
out = {"root": cfg.get("root", "data/pano"), "panoramas": [dict(c[3], edits=[c[4]]) for c in pick]}
with open(a.out, "w", encoding="utf-8") as f:
    yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, width=200)
for gap, gi, gs, p, e in pick:
    print(f"{p['image']:28s} {e['name']:10s} independent gain {gi:+.2f}   sync gain {gs:+.2f}")
