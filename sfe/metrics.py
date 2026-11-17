"""Panorama editing metrics.

- clip_views:  CLIP score of the target prompt on perspective views of the edited ERP
- lpips_views: LPIPS between source / edited perspective views (structure preservation)
- seam_ratio:  |col_0 - col_{W-1}| / mean |col_j - col_{j+1}|; ~1 = seamless, >>1 = visible seam
- view_disagreement: weighted std of overlapping view predictions before blending
                     (0 for perfectly consistent views; large for independent per-view edits)
Perspective metrics are computed on view_set("eval8") + poles, which differ from the editing views.
"""
from __future__ import annotations

import torch

from .geometry import PanoProjector, View, view_set

_clip = None
_lpips = None


def eval_views(fov=90.0):
    return view_set("eval8", fov) + [View(0, 90, fov), View(0, -90, fov)]


def perspective(erp: torch.Tensor, px=512, views=None) -> torch.Tensor:
    """(1, 3, H, W) -> (N, 3, px, px)."""
    views = views or eval_views()
    proj = PanoProjector(views, erp.shape[-2:], (px, px), device=erp.device)
    return proj.to_views(erp[0].float())


@torch.no_grad()
def clip_score(imgs: torch.Tensor, prompt: str, model="openai/clip-vit-large-patch14") -> torch.Tensor:
    global _clip
    from transformers import CLIPModel, CLIPProcessor

    if _clip is None:
        _clip = (CLIPModel.from_pretrained(model).to(imgs.device).eval(), CLIPProcessor.from_pretrained(model))
    m, proc = _clip
    pix = ((imgs.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    batch = proc(text=[prompt], images=list(pix), return_tensors="pt", padding=True).to(imgs.device)
    out = m(**batch)
    i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return 100 * (i @ t.T).squeeze(-1)  # (N,)


@torch.no_grad()
def lpips_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _lpips
    import lpips

    if _lpips is None:
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(a.device).eval()
    return _lpips(a.float(), b.float()).flatten()


def seam_ratio(erp: torch.Tensor) -> float:
    x = erp.float()
    seam = (x[..., 0] - x[..., -1]).abs().mean()
    inner = (x[..., 1:] - x[..., :-1]).abs().mean()
    return (seam / inner.clamp_min(1e-8)).item()


def view_disagreement(views_px: torch.Tensor, proj: PanoProjector) -> float:
    num, den = proj.splat(views_px)
    num2, _ = proj.splat(views_px.float() ** 2)
    mean = num / den.clamp_min(1e-8)
    var = (num2 / den.clamp_min(1e-8) - mean**2).clamp_min(0)
    overlap = (proj.bwd_weight > 0).sum(0) > 1
    return var.mean(0, keepdim=True)[overlap].sqrt().mean().item()


def evaluate(src_erp: torch.Tensor, out_erp: torch.Tensor, tar_prompt: str, px=512) -> dict:
    vs, vo = perspective(src_erp, px), perspective(out_erp, px)
    return {
        "clip_tar": clip_score(vo, tar_prompt).mean().item(),
        "lpips_src": lpips_dist(vs, vo).mean().item(),
        "seam_ratio": seam_ratio(out_erp),
        "seam_ratio_src": seam_ratio(src_erp),
    }
