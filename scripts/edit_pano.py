"""Edit one 360-degree panorama.

python scripts/edit_pano.py --image data/pano/xxx.jpg --src "a living room" --tar "a living room in winter, snow" \
    --model flux --method sync --out outputs/demo
methods: sync (ours) | independent (per-view FlowEdit, then blend) | erp (FlowEdit on the whole ERP)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3")  # our server's GPU 0 is faulty, so default to GPUs 1-3; override with CUDA_VISIBLE_DEVICES

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sfe.models import DEFAULTS, load_backend, resize, to_pil, to_tensor  # noqa: E402
from sfe.sync_flowedit import EditParams, SyncConfig, erp_flowedit, sync_flowedit  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--src", required=True, help="source prompt")
    ap.add_argument("--tar", required=True, help="target prompt")
    ap.add_argument("--model", default="flux", choices=["sd3", "flux"])
    ap.add_argument("--method", default="sync", choices=["sync", "independent", "erp"])
    ap.add_argument("--erp_h", type=int, default=1024, help="ERP height; width = 2x")
    ap.add_argument("--views", default="dense18")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--view_px", type=int, default=768)
    ap.add_argument("--weight", default="cosine")
    ap.add_argument("--weight_pow", type=float, default=1.0)
    ap.add_argument("--rotate", action="store_true", help="random rotation of the view layout every step")
    ap.add_argument("--pitch_jitter", type=float, default=15.0)
    ap.add_argument("--out_views", default="dense18")
    ap.add_argument("--decode_views", default="", help="diagnostic: decode with another layout")
    ap.add_argument("--dv_rescale", default="none", choices=["none", "global", "local"])
    ap.add_argument("--rescale_max", type=float, default=3.0)
    ap.add_argument("--rescale_steps", default="all", choices=["all", "early", "late"])
    ap.add_argument("--view_prompts", default="none", choices=["none", "hint", "ground"])
    ap.add_argument("--noise", default="view", choices=["view", "pano"])
    ap.add_argument("--src_mode", default="encode", choices=["encode", "project"])
    ap.add_argument("--output", default="views", choices=["views", "erp"])
    ap.add_argument("--T", type=int)
    ap.add_argument("--n_max", type=int)
    ap.add_argument("--n_avg", type=int)
    ap.add_argument("--g_src", type=float)
    ap.add_argument("--g_tar", type=float)
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--eval", action="store_true", help="compute CLIP / LPIPS / seam metrics")
    return ap


def params_from_args(args) -> EditParams:
    d = dict(DEFAULTS[args.model])
    for k in ["T", "n_max", "n_avg", "g_src", "g_tar"]:
        if getattr(args, k) is not None:
            d[k] = getattr(args, k)
    return EditParams(**d)


def run(backend, args, img: torch.Tensor):
    p = params_from_args(args)
    g = torch.Generator(device=backend.device).manual_seed(args.seed)
    if args.method == "erp":
        return erp_flowedit(backend, img, args.src, args.tar, p, g), {}
    cfg = SyncConfig(views=args.views, fov=args.fov, view_px=args.view_px, weight=args.weight, weight_pow=args.weight_pow,
                     rotate=args.rotate, pitch_jitter=args.pitch_jitter, out_views=args.out_views, decode_views=args.decode_views, dv_rescale=args.dv_rescale, rescale_max=args.rescale_max, rescale_steps=args.rescale_steps, view_prompts=args.view_prompts,
                     sync="dv" if args.method == "sync" else "none", noise=args.noise,
                     src_mode=args.src_mode, output=args.output)
    return sync_flowedit(backend, img, args.src, args.tar, p, cfg, g)


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    backend = load_backend(args.model)
    backend.max_batch = args.max_batch
    img = resize(to_tensor(Image.open(args.image)), (args.erp_h, 2 * args.erp_h))
    t0 = time.time()
    out, extra = run(backend, args, img)
    dt = time.time() - t0
    stem = os.path.splitext(os.path.basename(args.image))[0]
    to_pil(out).save(os.path.join(args.out, f"{stem}_{args.method}.png"))
    info = {"seconds": dt, **vars(args)}
    if args.eval:
        from sfe.metrics import evaluate

        info.update(evaluate(img.to(out.device), out, args.tar))
    with open(os.path.join(args.out, f"{stem}_{args.method}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
