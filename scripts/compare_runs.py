"""Compare several result folders on the same (image, edit) subset.
python scripts/compare_runs.py --config configs/val8.yaml base=outputs/bench/flux:sync pow4=outputs/val/pow4/flux:sync ...
Each argument is name=folder:method. CLIP gain is measured against the unedited source
(outputs/bench/flux_source_clip.json, made after the first benchmark).
"""
import argparse
import json
import os
import statistics as st

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/val8.yaml")
ap.add_argument("--source_clip", default="outputs/bench/flux_source_clip.json")
ap.add_argument("runs", nargs="+")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
keys = [(p["image"], e["name"]) for p in cfg["panoramas"] for e in p["edits"]]
base = json.load(open(a.source_clip))
print(f"| run | n | CLIP ↑ | CLIP gain ↑ | gain>0.5 | LPIPS ↓ | seam dev ↓ | sec |")
print("|---|---|---|---|---|---|---|---|")
for spec in a.runs:
    name, rest = spec.split("=", 1)
    folder, method = rest.rsplit(":", 1)
    rs = []
    for img, ed in keys:
        f = os.path.join(folder, f"{img[:-4]}__{ed}__{method}.json")
        if os.path.exists(f):
            r = json.load(open(f))
            r["gain"] = r["clip_tar"] - base[f"{img}|{ed}"]
            rs.append(r)
    if not rs:
        print(f"| {name} | 0 | - | - | - | - | - | - |")
        continue
    m = lambda k: st.mean(r[k] for r in rs)
    seam = st.mean(abs(r["seam_ratio"] - r["seam_ratio_src"]) for r in rs)
    print(f"| {name} | {len(rs)} | {m('clip_tar'):.2f} | {m('gain'):+.2f} | {sum(r['gain'] > 0.5 for r in rs)}/{len(rs)} "
          f"| {m('lpips_src'):.3f} | {seam:.3f} | {m('seconds'):.0f} |")
