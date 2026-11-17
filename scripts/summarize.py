"""Average the per-job JSON metrics into method x metric tables (Markdown), overall and per group.
python scripts/summarize.py outputs/bench/flux
"""
import glob
import json
import os
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "outputs/bench/flux"
recs = [json.load(open(f)) for f in glob.glob(os.path.join(root, "*.json"))]
keys = ["clip_tar", "lpips_src", "seam_ratio", "seconds"]
better = {"clip_tar": "↑", "lpips_src": "↓", "seam_ratio": "≈1", "seconds": "↓"}


def table(rs, title):
    rows = defaultdict(lambda: defaultdict(list))
    for r in rs:
        for k in keys:
            if isinstance(r.get(k), (int, float)):
                rows[r["method"]][k].append(r[k])
    print(f"\n### {title}\n")
    print("| method | n | " + " | ".join(f"{k} {better[k]}" for k in keys) + " |")
    print("|---" * (len(keys) + 2) + "|")
    for m in ["erp", "independent", "sync"] + sorted(set(rows) - {"erp", "independent", "sync"}):
        if m not in rows:
            continue
        d = rows[m]
        n = len(d[keys[0]])
        print(f"| {m} | {n} | " + " | ".join(f"{sum(d[k]) / len(d[k]):.4f}" if d[k] else "-" for k in keys) + " |")


table(recs, f"all ({root})")
for g in sorted({r.get("group", "") for r in recs} - {""}):
    table([r for r in recs if r.get("group") == g], g)
src = [r["seam_ratio_src"] for r in recs if "seam_ratio_src" in r]
if src:
    print(f"\nsource panoramas seam ratio: {sum(src) / len(src):.4f}")
