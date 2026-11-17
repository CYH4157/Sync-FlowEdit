"""FlowEdit (Kulikov et al., ICCV 2025), re-implemented from Algorithm 1 of the paper.

The edit ODE runs on Z_edit, starting from Z_edit = X_src at noise level s_{n_max}:
    Z_src(s) = (1 - s) X_src + s eps
    Z_tar(s) = Z_edit + Z_src(s) - X_src
    dZ_edit  = (s_next - s) * [v(Z_tar, s, c_tar) - v(Z_src, s, c_src)]   (averaged over n_avg noises)
For the last n_min steps it switches to plain target-prompt sampling (SDEdit-like).

Written from the paper, not copied from the official repo (github.com/fallenshock/FlowEdit, MIT).
"""
from __future__ import annotations

import torch

from .models import Cond, FlowBackend


def active_steps(T: int, n_max: int):
    """Indices i (into sigmas) that FlowEdit actually integrates: the last n_max of T."""
    return range(T - n_max, T)


@torch.no_grad()
def flowedit(
    backend: FlowBackend,
    x_src: torch.Tensor,
    c_src: Cond,
    c_tar: Cond,
    T: int = 50,
    n_max: int = 33,
    n_min: int = 0,
    n_avg: int = 1,
    g_src: float = 3.5,
    g_tar: float = 13.5,
    generator: torch.Generator | None = None,
    callback=None,
) -> torch.Tensor:
    """x_src: clean source latent (B, C, h, w). Returns the edited latent."""
    sig = backend.sigmas(T, x_src.shape[-2:])
    x_src = x_src.float()
    z_edit = x_src.clone()
    z_tar = None
    for i in active_steps(T, n_max):
        s, s_next = sig[i], sig[i + 1]
        remaining = T - i
        if remaining > n_min:
            dv = torch.zeros_like(x_src)
            for _ in range(n_avg):
                eps = torch.randn(x_src.shape, generator=generator, device=x_src.device)
                z_src = (1 - s) * x_src + s * eps
                zt = z_edit + z_src - x_src
                v_src, v_tar = backend.velocity([z_src, zt], [c_src, c_tar], [g_src, g_tar], s)
                dv += (v_tar - v_src) / n_avg
            z_edit = z_edit + (s_next - s) * dv
        else:
            if z_tar is None:  # enter the SDEdit-like regime once
                eps = torch.randn(x_src.shape, generator=generator, device=x_src.device)
                z_tar = z_edit + s * (eps - x_src)  # = Z_edit + Z_src(s) - X_src
            (v_tar,) = backend.velocity([z_tar], [c_tar], [g_tar], s)
            z_tar = z_tar + (s_next - s) * v_tar
        if callback is not None:
            callback(i, z_edit if z_tar is None else z_tar)
    return z_edit if z_tar is None else z_tar
