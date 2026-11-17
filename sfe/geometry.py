"""Equirectangular (ERP) <-> perspective projection utilities.

Conventions
-----------
World frame: +x right, +y up, +z forward (yaw = 0, pitch = 0 looks along +z).
ERP pixel (row i, col j) of an H x W image has its center at
    lon = ((j + 0.5) / W * 2 - 1) * pi        in (-pi, pi)
    lat = (1 - (i + 0.5) / H * 2) * pi / 2    in (-pi/2, pi/2), top row = north.
A perspective view is a pinhole camera with horizontal field of view `fov`.

All sampling uses torch.nn.functional.grid_sample (align_corners=False), so
the same projector works for pixels (3 channels) and VAE latents (16 channels)
as long as you pass the matching resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class View:
    yaw: float  # degrees, positive turns right
    pitch: float  # degrees, positive looks up
    fov: float = 90.0  # horizontal field of view, degrees


def view_set(name: str, fov: float = 90.0) -> list[View]:
    """Predefined camera layouts that cover the whole sphere."""
    if name == "ring8":  # equator only (poles NOT covered; for ablation)
        return [View(45.0 * k, 0.0, fov) for k in range(8)]
    if name == "ring8_poles":  # 8 equator + 2 poles = 10 views
        return [View(45.0 * k, 0.0, fov) for k in range(8)] + [
            View(0.0, 90.0, fov),
            View(0.0, -90.0, fov),
        ]
    if name == "dense18":  # 8 equator + 4 up + 4 down + 2 poles
        vs = [View(45.0 * k, 0.0, fov) for k in range(8)]
        vs += [View(45.0 + 90.0 * k, 50.0, fov) for k in range(4)]
        vs += [View(45.0 + 90.0 * k, -50.0, fov) for k in range(4)]
        vs += [View(0.0, 90.0, fov), View(0.0, -90.0, fov)]
        return vs
    if name == "cube6":
        return [View(90.0 * k, 0.0, 90.0) for k in range(4)] + [
            View(0.0, 90.0, 90.0),
            View(0.0, -90.0, 90.0),
        ]
    if name == "eval8":  # evaluation views, offset from the editing ring
        return [View(22.5 + 45.0 * k, 0.0, fov) for k in range(8)]
    raise ValueError(f"unknown view set {name!r}")


def rotation(yaw_deg: float, pitch_deg: float) -> torch.Tensor:
    """Camera-to-world rotation R = R_yaw @ R_pitch (float64, 3x3)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    r_yaw = torch.tensor(
        [[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]],
        dtype=torch.float64,
    )
    r_pitch = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, math.cos(p), math.sin(p)], [0.0, -math.sin(p), math.cos(p)]],
        dtype=torch.float64,
    )
    return r_yaw @ r_pitch


def _pixel_centers(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float64) + 0.5) / n * 2.0 - 1.0


def erp_directions(H: int, W: int) -> torch.Tensor:
    """Unit world directions of ERP pixel centers, shape (H, W, 3)."""
    lon = _pixel_centers(W) * math.pi
    lat = -_pixel_centers(H) * (math.pi / 2)
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return torch.stack(
        [torch.cos(lat) * torch.sin(lon), torch.sin(lat), torch.cos(lat) * torch.cos(lon)], dim=-1
    )


def perspective_directions(view: View, h: int, w: int) -> torch.Tensor:
    """Unit world directions of perspective pixel centers, shape (h, w, 3)."""
    tan_x = math.tan(math.radians(view.fov) / 2)
    tan_y = tan_x * h / w
    yy, xx = torch.meshgrid(_pixel_centers(h), _pixel_centers(w), indexing="ij")
    d = torch.stack([xx * tan_x, -yy * tan_y, torch.ones_like(xx)], dim=-1)
    d = d / d.norm(dim=-1, keepdim=True)
    return d @ rotation(view.yaw, view.pitch).T


def _erp_grid_from_dirs(d: torch.Tensor, W: int) -> torch.Tensor:
    """Directions -> grid_sample coords on a horizontally wrap-padded ERP (W+2 wide)."""
    lon = torch.atan2(d[..., 0], d[..., 2])
    lat = torch.asin(d[..., 1].clamp(-1.0, 1.0))
    u = lon / math.pi  # normalized x on the original ERP
    v = -lat / (math.pi / 2)
    # account for one wrapped column on each side: x_pix' = x_pix + 1
    x_pix = (u + 1.0) / 2.0 * W - 0.5
    u_pad = (x_pix + 1.0 + 0.5) / (W + 2) * 2.0 - 1.0
    return torch.stack([u_pad, v], dim=-1)


def wrap_pad(erp: torch.Tensor) -> torch.Tensor:
    """Circularly pad one column on both sides (..., H, W) -> (..., H, W+2)."""
    return torch.cat([erp[..., -1:], erp, erp[..., :1]], dim=-1)


class PanoProjector:
    """Precomputed forward (ERP -> views) and backward (views -> ERP) warps.

    weight: "cosine" gives a smooth window that falls to 0 at view borders
    (seamless blending); "uniform" is a hard box (ablation). weight_pow sharpens the
    cosine window (less averaging between overlapping views).
    """

    def __init__(
        self,
        views: Sequence[View],
        erp_hw: tuple[int, int],
        view_hw: tuple[int, int],
        weight: str = "cosine",
        device: torch.device | str = "cpu",
        weight_pow: float = 1.0,
        backward: bool = True,
    ):
        self.views = list(views)
        self.H, self.W = erp_hw
        self.h, self.w = view_hw
        self.device = torch.device(device)

        fwd = [_erp_grid_from_dirs(perspective_directions(v, self.h, self.w), self.W) for v in self.views]
        self.fwd_grid = torch.stack(fwd).float().to(self.device)  # (N, h, w, 2)

        if not backward:  # forward warp only (to_views); to_erp / splat unavailable
            self.bwd_grid = self.bwd_weight = self.weight_sum = None
            return
        d_world = erp_directions(self.H, self.W)  # (H, W, 3)
        grids, weights = [], []
        for v in self.views:
            tan_x = math.tan(math.radians(v.fov) / 2)
            tan_y = tan_x * self.h / self.w
            d_cam = d_world @ rotation(v.yaw, v.pitch)  # = R^T d
            z = d_cam[..., 2]
            zs = z.clamp_min(1e-6)
            xn = d_cam[..., 0] / (zs * tan_x)
            yn = -d_cam[..., 1] / (zs * tan_y)
            valid = (z > 1e-6) & (xn.abs() < 1) & (yn.abs() < 1)
            if weight == "cosine":
                wgt = torch.cos(xn.clamp(-1, 1) * math.pi / 2) * torch.cos(yn.clamp(-1, 1) * math.pi / 2)
                wgt = wgt.clamp_min(0) ** weight_pow  # >1: each point is dominated by its most central view
            elif weight == "uniform":
                wgt = torch.ones_like(xn)
            else:
                raise ValueError(weight)
            grids.append(torch.stack([xn, yn], dim=-1))
            weights.append(torch.where(valid, wgt, torch.zeros_like(wgt)))
        self.bwd_grid = torch.stack(grids).float().to(self.device)  # (N, H, W, 2)
        self.bwd_weight = torch.stack(weights).float().unsqueeze(1).to(self.device)  # (N,1,H,W)
        self.weight_sum = self.bwd_weight.sum(0)  # (1, H, W)

    @property
    def coverage(self) -> float:
        """Fraction of ERP pixels seen (weight > 0) by at least one view."""
        return (self.weight_sum > 0).float().mean().item()

    def to_views(self, erp: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """(C, H, W) -> (N, C, h, w)."""
        assert erp.shape[-2:] == (self.H, self.W), erp.shape
        src = wrap_pad(erp.float()).unsqueeze(0).expand(len(self.views), -1, -1, -1)
        out = F.grid_sample(src, self.fwd_grid, mode=mode, padding_mode="border", align_corners=False)
        return out.to(erp.dtype)

    def splat(self, views: torch.Tensor, mode: str = "bilinear") -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-warp every view into ERP space; returns (weighted sum, weight sum)."""
        assert views.shape[0] == len(self.views) and views.shape[-2:] == (self.h, self.w), views.shape
        samp = F.grid_sample(views.float(), self.bwd_grid, mode=mode, padding_mode="border", align_corners=False)
        return (samp * self.bwd_weight).sum(0), self.weight_sum

    def to_erp(self, views: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
        """(N, C, h, w) -> (C, H, W) weighted average. Uncovered pixels take `fallback` (or 0)."""
        num, den = self.splat(views)
        out = num / den.clamp_min(1e-8)
        if fallback is not None:
            out = torch.where(den > 0, out, fallback.float())
        return out.to(views.dtype)
