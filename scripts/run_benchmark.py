"""Run every (panorama, edit, method) in a YAML file, sharded over GPUs.

One process per GPU (3x H100; our server's GPU 0 is faulty, so rank r uses GPU r+1):
    for r in 0 1 2; do CUDA_VISIBLE_DEVICES=$((r+1)) python scripts/run_benchmark.py \
        --config configs/edits.yaml --rank $r --world 3 --model flux --out outputs/bench & done; wait
    python scripts/summarize.py outputs/bench
Finished jobs are skipped, so the command can be re-run after a crash.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3")  # our server's GPU 0 is faulty, so default to GPUs 1-3; override with CUDA_VISIBLE_DEVICES

import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from edit_pano import build_parser, run  # noqa: E402
from sfe.metrics import evaluate  # noqa: E402
from sfe.models import load_backend, resize, to_pil, to_tensor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/edits.yaml")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--model", default="flux")
    ap.add_argument("--methods", default="sync,independent,erp")
    ap.add_argument("--out", default="outputs/bench")
    ap.add_argument("--extra", default="", help="extra edit_pano flags, e.g. '--noise pano --view_px 1024'")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    # all workers walk the same list and claim jobs with an atomic lock file (dynamic load balancing);
    # --rank only rotates the starting point. Delete *.lock files after a crash before re-running.
    jobs = [(item, e, m) for item in cfg["panoramas"] for e in item["edits"] for m in a.methods.split(",")]
    k = a.rank * len(jobs) // max(a.world, 1)
    jobs = jobs[k:] + jobs[:k]
    backend = load_backend(a.model)
    root = cfg.get("root", ".")
    out_dir = os.path.join(a.out, a.model)
    os.makedirs(out_dir, exist_ok=True)
    for item, e, m in jobs:
        tag = f"{os.path.splitext(item['image'])[0]}__{e['name']}__{m}"
        if os.path.exists(os.path.join(out_dir, tag + ".json")):
            continue
        try:
            os.close(os.open(os.path.join(out_dir, tag + ".lock"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue  # another worker has it
        try:
            args = build_parser().parse_args(
                ["--image", os.path.join(root, item["image"]), "--src", item["src"], "--tar", e["tar"],
                 "--model", a.model, "--method", m] + a.extra.split()
            )
            backend.max_batch = args.max_batch
            img = resize(to_tensor(Image.open(args.image)), (args.erp_h, 2 * args.erp_h))
            t0 = time.time()
            out, _ = run(backend, args, img)
            dt = time.time() - t0
            to_pil(out).save(os.path.join(out_dir, tag + ".png"))
            res = {"image": item["image"], "group": item.get("group", ""), "edit": e["name"], "method": m,
                   "seconds": dt, **evaluate(img.to(out.device), out, e["tar"])}
            json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
            print(f"[rank {a.rank}] {tag}: {res}", flush=True)
        except Exception as ex:  # keep going; the failure is logged and the job can be retried
            print(f"[rank {a.rank}] FAILED {tag}: {type(ex).__name__}: {ex}", flush=True)
            with open(os.path.join(out_dir, "failed.txt"), "a") as f:
                f.write(f"{tag}\t{type(ex).__name__}: {ex}\n")
        finally:
            if os.path.exists(os.path.join(out_dir, tag + ".lock")):
                os.remove(os.path.join(out_dir, tag + ".lock"))
            torch.cuda.empty_cache()
    print(f"[rank {a.rank}] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
