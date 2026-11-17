"""Backend sanity checks on one GPU.

1. text-to-image sampling with backend.velocity + sigmas (is the model wrapper correct?)
2. FlowEdit on a perspective crop with an n_max sweep (how much structure is kept?)
python scripts/sanity_check.py --model flux --out outputs/sanity
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3")  # our server's GPU 0 is faulty, so default to GPUs 1-3; override with CUDA_VISIBLE_DEVICES

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sfe.flowedit import flowedit  # noqa: E402
from sfe.geometry import PanoProjector, View  # noqa: E402
from sfe.metrics import lpips_dist  # noqa: E402
from sfe.models import DEFAULTS, load_backend, resize, to_pil, to_tensor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="flux")
ap.add_argument("--out", default="outputs/sanity")
ap.add_argument("--pano", default="data/pano/hotel_room.jpg")
ap.add_argument("--yaw", type=float, default=-15)
ap.add_argument("--pitch", type=float, default=-15)
ap.add_argument("--src", default="a photo of a modern hotel room with a bed, a yellow armchair and a desk")
ap.add_argument("--tar", default="a photo of a modern hotel room with a bed, a red armchair and a desk")
ap.add_argument("--nmax", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
be = load_backend(a.model)
d = DEFAULTS[a.model]

# 1. plain sampling
g = torch.Generator(device=be.device).manual_seed(0)
z = torch.randn((1, 16, 128, 128), generator=g, device=be.device)
sig = be.sigmas(d["T"], (128, 128))
c = be.text(a.tar)
for i in range(d["T"]):
    (v,) = be.velocity([z], [c], [d["g_tar"] if a.model == "sd3" else 3.5], sig[i])
    z = z + (sig[i + 1] - sig[i]) * v
to_pil(be.decode(z)).save(f"{a.out}/{a.model}_t2i.png")
print("t2i saved", flush=True)

# 2. FlowEdit n_max sweep on a centred crop
img = resize(to_tensor(Image.open(a.pano)), (2048, 4096))
view = PanoProjector([View(a.yaw, a.pitch, 90)], (2048, 4096), (1024, 1024)).to_views(img[0]).to(be.device)
to_pil(view).save(f"{a.out}/{a.model}_src.png")
x = be.encode(view).float()
rec = be.decode(x).float()
print(f"VAE recon LPIPS {lpips_dist(rec, view.float()).item():.3f}", flush=True)
cs, ct = be.text(a.src), be.text(a.tar)
sweep = [int(n) for n in a.nmax.split(",")] if a.nmax else ([24, 20, 16] if a.model == "flux" else [33, 28, 22])
for n in sweep:
    p = dict(d, n_max=n)
    g = torch.Generator(device=be.device).manual_seed(0)
    out = be.decode(flowedit(be, x, cs, ct, generator=g, **p)).float()
    to_pil(out).save(f"{a.out}/{a.model}_nmax{n}.png")
    print(f"n_max={n}: LPIPS to source {lpips_dist(out, view.float()).item():.3f}", flush=True)
