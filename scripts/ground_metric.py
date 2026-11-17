"""Does sky content (sun, glow) leak onto the ground?
python scripts/ground_metric.py --config configs/sky8.yaml base=outputs/bench_resc/flux:sync hint=outputs/val/hint/flux:sync
ground_bright = area fraction of the ground band (latitude < -40 deg, cos-lat weighted) whose luminance rose by > --thr
ground_dY     = mean luminance change of that band (night edits should darken it, not brighten it)
sky_dY        = the same for the sky band (latitude > +40 deg), for reference: the edit should happen up there
"""
import argparse
import math
import os

import numpy as np
import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sky8.yaml")
ap.add_argument("--thr", type=float, default=0.2)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
root = cfg.get("root", "data/pano")
keys = [(p["image"], e["name"]) for p in cfg["panoramas"] for e in p["edits"]]
H, W = a.size, 2 * a.size
lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
wcol = np.cos(np.radians(lat))[:, None].repeat(W, 1)
ground, sky = lat[:, None].repeat(W, 1) < -40, lat[:, None].repeat(W, 1) > 40


def lum(path):
    x = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def band_mean(v, m):
    return float((v * wcol * m).sum() / (wcol * m).sum())


src = {img: lum(os.path.join(root, img)) for img, _ in keys}
print("| run | n | ground_bright ↓ | ground_dY | sky_dY |")
print("|---|---|---|---|---|")
for spec in a.runs:
    name, rest = spec.split("=", 1)
    folder, method = rest.rsplit(":", 1)
    gb, gd, sd = [], [], []
    for img, ed in keys:
        f = os.path.join(folder, f"{img[:-4]}__{ed}__{method}.png")
        if not os.path.exists(f):
            continue
        d = lum(f) - src[img]
        gb.append(band_mean((d > a.thr).astype(np.float32), ground))
        gd.append(band_mean(d, ground))
        sd.append(band_mean(d, sky))
    if gb:
        print(f"| {name} | {len(gb)} | {np.mean(gb):.3f} | {np.mean(gd):+.3f} | {np.mean(sd):+.3f} |")
