"""Sanity check: plain FlowEdit on an ordinary photo (compare against the official HF demo).
python scripts/edit_image.py --image cat.png --src "a cat sitting on grass" --tar "a dog sitting on grass" --model sd3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3")  # our server's GPU 0 is faulty, so default to GPUs 1-3; override with CUDA_VISIBLE_DEVICES

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sfe.flowedit import flowedit  # noqa: E402
from sfe.models import DEFAULTS, load_backend, resize, to_pil, to_tensor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--tar", required=True)
ap.add_argument("--model", default="sd3", choices=["sd3", "flux"])
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/single.png")
a = ap.parse_args()

be = load_backend(a.model)
img = resize(to_tensor(Image.open(a.image)), (a.size, a.size)).to(be.device)
g = torch.Generator(device=be.device).manual_seed(a.seed)
z = flowedit(be, be.encode(img).float(), be.text(a.src), be.text(a.tar), generator=g, **DEFAULTS[a.model])
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
to_pil(be.decode(z)).save(a.out)
print("saved", a.out)
