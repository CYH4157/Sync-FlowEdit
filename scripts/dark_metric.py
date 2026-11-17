"""Quantify dark smears: fraction of the sphere that got clearly darker than the source.
python scripts/dark_metric.py --config configs/val8.yaml base=outputs/bench/flux:sync g_max2=outputs/val/g_max2/flux:sync
dark_frac  = area fraction (cos-latitude weighted) whose luminance dropped by more than --thr (0..1 scale)
light_frac = same for brightening (snow / ice edits should brighten)
Edits that legitimately darken (names in --skip, e.g. candle at night) are excluded from the mean.
"""
import argparse
import math
import os

import numpy as np
import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/val8.yaml")
ap.add_argument("--thr", type=float, default=0.15)
ap.add_argument("--skip", default="candle,night,neon,horror,fog")
ap.add_argument("--size", type=int, default=512, help="ERP height used for the measurement")
ap.add_argument("runs", nargs="+")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
root = cfg.get("root", "data/pano")
skip = set(a.skip.split(","))
keys = [(p["image"], e["name"]) for p in cfg["panoramas"] for e in p["edits"] if e["name"] not in skip]
H, W = a.size, 2 * a.size
lat = (0.5 - (np.arange(H) + 0.5) / H) * math.pi
area = np.repeat(np.cos(lat)[:, None], W, axis=1)
area /= area.sum()


def lum(path):
    x = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


src = {img: lum(os.path.join(root, img)) for img, _ in keys}
print(f"edits used: {len(keys)} (skipped: {sorted(skip)})")
print("| run | dark_frac ↓ | light_frac | mean ΔY |")
print("|---|---|---|---|")
for spec in a.runs:
    name, rest = spec.split("=", 1)
    folder, method = rest.rsplit(":", 1)
    d, l, m = [], [], []
    for img, ed in keys:
        f = os.path.join(folder, f"{img[:-4]}__{ed}__{method}.png")
        if not os.path.exists(f):
            continue
        diff = lum(f) - src[img]
        d.append(float((area * (diff < -a.thr)).sum()))
        l.append(float((area * (diff > a.thr)).sum()))
        m.append(float((area * diff).sum()))
    if d:
        print(f"| {name} | {np.mean(d):.3f} | {np.mean(l):.3f} | {np.mean(m):+.3f} |")
    else:
        print(f"| {name} | - | - | - |")
