"""Sync-FlowEdit: training-free, seam-free text editing of 360-degree panoramas.

Idea: FlowEdit's ODE only evolves the *edit offset* D = Z_edit - X_src (D(s_{n_max}) = 0):
    dD = (s_next - s) * [v(Z_src_i + D_i, c_tar) - v(Z_src_i, c_src)]
We keep ONE offset D in the shared ERP latent canvas and, at every step,
  1. project it into N perspective views:     D_i = P_i(D)
  2. compute FlowEdit's velocity difference per view (perspective views are in-distribution
     for a text-to-image model, the ERP image is not),
  3. splat the N differences back with a smooth window and average (synchronization),
  4. take an Euler step on D.
Because the canvas is a sphere there is no left/right seam and the poles are seen by
undistorted top/bottom cameras. sync="none" gives the "independent views" baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .flowedit import active_steps, flowedit
from .geometry import PanoProjector, View, WarpedNoise, view_set
from .models import FlowBackend
from .prompts import ground_prompt


@dataclass
class EditParams:
    T: int = 50
    n_max: int = 33
    n_min: int = 0
    n_avg: int = 1
    g_src: float = 3.5
    g_tar: float = 13.5


@dataclass
class SyncConfig:
    views: str = "dense18"  # see geometry.view_set
    fov: float = 90.0
    view_px: int = 1024  # perspective view resolution (pixels)
    weight: str = "cosine"  # splat window: cosine | uniform
    weight_pow: float = 1.0  # cosine window exponent (>1 = less overlap averaging)
    sync: str = "dv"  # dv: synchronize velocity differences (ours) | none: independent views
    noise: str = "view"  # view: i.i.d. noise per view | pano: one ERP noise, projected (nearest) | warp: Gaussian-preserving warp
    noise_up: int = 8  # with noise=warp: fine noise grid = noise_up x the ERP latent resolution
    src_mode: str = "encode"  # encode: VAE-encode each perspective crop | project: warp the ERP latent
    output: str = "views"  # views: decode views, blend in pixel space | erp: circular-decode ERP latent
    rotate: bool = False  # re-draw a random rotation of the view layout at every step (moving boundaries)
    pitch_jitter: float = 15.0  # with rotate: random pitch offset (deg) for non-pole views
    out_views: str = "dense18"  # with rotate: fixed dense layout used only for the final decode/blend
    rotate_seed: int = 0
    dv_rescale: str = "none"  # none | global | local: undo the magnitude lost when averaging views
    rescale_max: float = 3.0
    rescale_steps: str = "all"  # all | early | late: which half of the edit steps gets the rescale
    decode_views: str = ""  # decode/blend with this layout instead (diagnostic); default: same as editing
    view_prompts: str = "none"  # none | hint (direction hint) | ground (down views: sky content removed by rules)
    hint_down: str = ", view looking down at the ground"
    hint_up: str = ", view looking up"
    hint_pitch: float = 45.0  # |pitch| at or beyond which a view counts as looking up / down
    extras: dict = field(default_factory=dict)


def _encode_views(backend: FlowBackend, views_px: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    return torch.cat([backend.encode(views_px[i : i + chunk]) for i in range(0, len(views_px), chunk)]).float()


def _decode_views(backend: FlowBackend, lat: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    return torch.cat([backend.decode(lat[i : i + chunk]).float() for i in range(0, len(lat), chunk)])


def _sync_dv(proj: PanoProjector, dv: torch.Tensor, cfg: SyncConfig, rescale: bool = True) -> torch.Tensor:
    """Weighted average of per-view velocity differences on the ERP canvas.

    Averaging views that disagree keeps the shared direction but shrinks its length. With
    dv_rescale, the average is scaled by (mean per-view |dv|) / |mean dv| (clipped to
    [1, rescale_max]), globally or per pixel (smoothed), so agreement is kept but strength is not lost.
    """
    num, den = proj.splat(dv)
    covered = den > 0
    avg = torch.where(covered, num / den.clamp_min(1e-8), torch.zeros_like(num))
    if cfg.dv_rescale == "none" or not rescale:
        return avg
    nrm_num, _ = proj.splat(dv.float().norm(dim=1, keepdim=True))
    mean_norm = nrm_num / den.clamp_min(1e-8)  # (1, H, W): average length of the views' dv
    avg_norm = avg.norm(dim=0, keepdim=True)  # (1, H, W): length after averaging
    if cfg.dv_rescale == "global":
        ratio = mean_norm[covered].mean() / avg_norm[covered].mean().clamp_min(1e-8)
        ratio = ratio.clamp(1.0, cfg.rescale_max)
    elif cfg.dv_rescale == "local":
        ratio = (mean_norm / avg_norm.clamp_min(1e-6)).clamp(1.0, cfg.rescale_max)
        ratio = torch.cat([ratio[..., -4:], ratio, ratio[..., :4]], dim=-1)  # wrap before smoothing
        ratio = F.avg_pool2d(ratio[None], 9, stride=1, padding=(4, 0), count_include_pad=False)[0]
    else:
        raise ValueError(cfg.dv_rescale)
    return avg * ratio


def _view_category(v: View, cfg: SyncConfig) -> str:
    if v.pitch <= -cfg.hint_pitch:
        return "down"
    if v.pitch >= cfg.hint_pitch:
        return "up"
    return "mid"


def _velocity_pairs(backend, z_src, z_tar, views, texts, cfg, p, s):
    """v_src, v_tar for every view; with view prompts each view gets its own (src, tar) conditioning."""
    if cfg.view_prompts == "none":
        return backend.velocity([z_src, z_tar], [texts["mid"][0], texts["mid"][1]], [p.g_src, p.g_tar], s)
    cats = [_view_category(v, cfg) for v in views]
    groups = {c: [i for i, k in enumerate(cats) if k == c] for c in dict.fromkeys(cats)}
    zs, conds, gs = [], [], []
    for c, idx in groups.items():
        zs += [z_src[idx], z_tar[idx]]
        conds += [texts[c][0], texts[c][1]]
        gs += [p.g_src, p.g_tar]
    outs = backend.velocity(zs, conds, gs, s)
    v_src, v_tar = torch.empty_like(z_src, dtype=torch.float32), torch.empty_like(z_tar, dtype=torch.float32)
    for g, (c, idx) in enumerate(groups.items()):
        v_src[idx], v_tar[idx] = outs[2 * g].float(), outs[2 * g + 1].float()
    return v_src, v_tar


@torch.no_grad()
def sync_flowedit(
    backend: FlowBackend,
    erp_img: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    p: EditParams,
    cfg: SyncConfig,
    generator: torch.Generator | None = None,
):
    """erp_img: (1, 3, H, W) in [-1, 1], H x W = 1 x 2, divisible by 16.

    Returns (edited ERP image (1, 3, H, W), dict with per-view decoded images and the final offset).
    """
    if p.n_min:
        raise NotImplementedError("Sync-FlowEdit currently supports n_min = 0 only")
    if cfg.rotate and cfg.sync != "dv":
        raise ValueError("rotate needs the shared ERP offset (sync=dv)")
    dev = backend.device
    erp_img = erp_img.to(dev).float()
    H, W = erp_img.shape[-2:]
    f = backend.vae_scale
    vp = cfg.view_px
    base_views = view_set(cfg.views, cfg.fov)
    X_erp = backend.encode_erp(erp_img)[0].float() if (cfg.src_mode == "project" or cfg.output == "erp") else None

    def make(views, backward=True):
        """Projectors + clean source latents for one view layout."""
        pl = PanoProjector(views, (H // f, W // f), (vp // f, vp // f), cfg.weight, dev, cfg.weight_pow)
        pp = PanoProjector(views, (H, W), (vp, vp), cfg.weight, dev, cfg.weight_pow, backward=backward)
        if cfg.src_mode == "encode":
            xs = _encode_views(backend, pp.to_views(erp_img[0]))
        elif cfg.src_mode == "project":
            xs = pl.to_views(X_erp)
        else:
            raise ValueError(cfg.src_mode)
        return pl, pp, xs

    layout_rng = torch.Generator().manual_seed(cfg.rotate_seed)

    def rotated(views):
        dy = float(torch.rand(1, generator=layout_rng)) * 360.0
        dp = (float(torch.rand(1, generator=layout_rng)) * 2 - 1) * cfg.pitch_jitter
        return [View(v.yaw + dy, v.pitch if abs(v.pitch) >= 89 else v.pitch + dp, cfg.fov) for v in views]

    proj_lat, proj_px, x_src = make(base_views)
    lat_hw, view_lat_hw = (H // f, W // f), tuple(x_src.shape[-2:])
    warp_noise = None
    if cfg.noise == "warp":
        if cfg.rotate:
            raise ValueError("noise=warp is precomputed for a fixed layout (no rotate)")
        warp_noise = WarpedNoise(base_views, view_lat_hw, (lat_hw[0] * cfg.noise_up, lat_hw[1] * cfg.noise_up), dev)
        if warp_noise.min_cells < 1:
            raise ValueError(f"noise_up={cfg.noise_up} too small: some view pixels get no noise cell")
    texts = {"mid": (backend.text(src_prompt), backend.text(tar_prompt))}
    if cfg.view_prompts == "hint":
        texts["down"] = (backend.text(src_prompt + cfg.hint_down), backend.text(tar_prompt + cfg.hint_down))
        texts["up"] = (backend.text(src_prompt + cfg.hint_up), backend.text(tar_prompt + cfg.hint_up))
    elif cfg.view_prompts == "ground":
        texts["down"] = (backend.text(ground_prompt(src_prompt)), backend.text(ground_prompt(tar_prompt)))
        texts["up"] = texts["mid"]
    elif cfg.view_prompts != "none":
        raise ValueError(cfg.view_prompts)
    sig = backend.sigmas(p.T, x_src.shape[-2:])
    C = x_src.shape[1]

    D = torch.zeros(C, H // f, W // f, device=dev)  # shared offset on the ERP latent canvas
    D_views = torch.zeros_like(x_src)  # used only when sync == "none"

    steps = list(active_steps(p.T, p.n_max))
    for k, i in enumerate(steps):
        s, s_next = sig[i], sig[i + 1]
        late = k >= len(steps) // 2
        use_rescale = cfg.rescale_steps == "all" or (cfg.rescale_steps == "late") == late
        if cfg.rotate:  # new random layout every step: boundaries never stay in one place
            proj_lat, proj_px, x_src = make(rotated(base_views), backward=False)
        Dv = proj_lat.to_views(D) if cfg.sync == "dv" else D_views
        dv = torch.zeros_like(x_src)
        for _ in range(p.n_avg):
            if cfg.noise == "view":
                eps = torch.randn(x_src.shape, generator=generator, device=dev)
            elif cfg.noise == "warp":
                eps = warp_noise.sample(C, generator)
            elif cfg.noise == "pano":
                pano_eps = torch.randn((C, H // f, W // f), generator=generator, device=dev)
                eps = proj_lat.to_views(pano_eps, mode="nearest")
            else:
                raise ValueError(cfg.noise)
            z_src = (1 - s) * x_src + s * eps
            v_src, v_tar = _velocity_pairs(backend, z_src, z_src + Dv, proj_lat.views, texts, cfg, p, s)
            dv += (v_tar - v_src) / p.n_avg
        if cfg.sync == "dv":
            D = D + (s_next - s) * _sync_dv(proj_lat, dv, cfg, use_rescale)  # uncovered pixels get 0 (no edit this step)
        elif cfg.sync == "none":
            D_views = D_views + (s_next - s) * dv
        else:
            raise ValueError(cfg.sync)

    final_layout = cfg.decode_views or (cfg.out_views if cfg.rotate else "")
    if final_layout:  # decode with a fixed (dense) layout, different from the editing views
        proj_lat, proj_px, x_src = make(view_set(final_layout, 90.0))
    Dv = proj_lat.to_views(D) if cfg.sync == "dv" else D_views
    views_px = _decode_views(backend, x_src + Dv)
    if cfg.output == "views":
        out = proj_px.to_erp(views_px, fallback=erp_img[0])[None]
    elif cfg.output == "erp":
        if cfg.sync != "dv":
            raise ValueError("output=erp needs a shared ERP offset (sync=dv)")
        out = backend.decode_erp(X_erp[None] + D[None]).float()
    else:
        raise ValueError(cfg.output)
    return out, {"views": views_px, "D": D, "views_cfg": proj_lat.views}


@torch.no_grad()
def erp_flowedit(backend: FlowBackend, erp_img, src_prompt, tar_prompt, p: EditParams, generator=None):
    """Baseline 1: run FlowEdit directly on the whole equirectangular image."""
    x = backend.encode_erp(erp_img.to(backend.device).float()).float()
    z = flowedit(backend, x, backend.text(src_prompt), backend.text(tar_prompt), T=p.T, n_max=p.n_max,
                 n_min=p.n_min, n_avg=p.n_avg, g_src=p.g_src, g_tar=p.g_tar, generator=generator)
    return backend.decode_erp(z).float()
