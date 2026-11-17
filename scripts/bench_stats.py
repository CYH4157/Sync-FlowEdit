"""Benchmark means in the same form as the slides' table: CLIP gain over the unedited source,
LPIPS to the source, and seam deviation |seam_ratio - seam_ratio_src|.
python scripts/bench_stats.py outputs/bench/flux:erp outputs/bench/flux:independent outputs/bench/flux:sync outputs/bench_resc/flux:sync
Prints one JSON object per run (also the per-edit win rate of the last run vs the first sync run).
"""
import glob
import json
import os
import statistics as st
import sys

base = json.load(open("outputs/bench/flux_source_clip.json"))
runs = {}
for spec in sys.argv[1:]:
    folder, method = spec.rsplit(":", 1)
    recs = {}
    for f in glob.glob(os.path.join(folder, f"*__{method}.json")):
        r = json.load(open(f))
        recs[(r["image"], r["edit"])] = r
    runs[spec] = recs
    gains = [r["clip_tar"] - base[f"{k[0]}|{k[1]}"] for k, r in recs.items()]
    out = {"run": spec, "n": len(recs),
           "clip": st.mean(gains), "edited": sum(g > 0.5 for g in gains),
           "lpips": st.mean(r["lpips_src"] for r in recs.values()),
           "seam": st.mean(abs(r["seam_ratio"] - r["seam_ratio_src"]) for r in recs.values()),
           "sec": st.median(r["seconds"] for r in recs.values())}
    print(json.dumps(out))
keys = list(runs)
if len(keys) >= 2:
    a, b = runs[keys[-1]], runs[keys[-2]]
    common = set(a) & set(b)
    better_clip = sum(a[k]["clip_tar"] > b[k]["clip_tar"] for k in common)
    print(json.dumps({"compare": f"{keys[-1]} vs {keys[-2]}", "n": len(common), "higher_clip": better_clip}))
