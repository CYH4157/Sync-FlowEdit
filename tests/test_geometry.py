"""CPU sanity checks for the projection code: python -m pytest tests  (or python tests/test_geometry.py)."""
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from sfe.geometry import PanoProjector, View, WarpedNoise, erp_directions, perspective_directions, view_set  # noqa: E402


def smooth_sphere_field(H, W):
    """A smooth function of the 3D direction, so it is continuous across the seam and poles."""
    d = erp_directions(H, W)
    x, y, z = d.unbind(-1)
    return torch.stack([torch.sin(3 * x) + y, torch.cos(2 * z) * x, y * y - z]).float()


def test_center_pixel_points_forward():
    d = perspective_directions(View(0, 0, 90), 64, 64)
    c = d[31:33, 31:33].mean((0, 1))
    assert torch.allclose(c / c.norm(), torch.tensor([0, 0, 1.0], dtype=torch.float64), atol=1e-3)
    d = perspective_directions(View(90, 0, 90), 64, 64)
    c = d[31:33, 31:33].mean((0, 1))
    assert torch.allclose(c / c.norm(), torch.tensor([1.0, 0, 0], dtype=torch.float64), atol=1e-3)
    d = perspective_directions(View(0, 90, 90), 64, 64)
    c = d[31:33, 31:33].mean((0, 1))
    assert torch.allclose(c / c.norm(), torch.tensor([0, 1.0, 0], dtype=torch.float64), atol=1e-3)


def test_coverage():
    assert PanoProjector(view_set("dense18"), (64, 128), (32, 32)).coverage == 1.0
    assert PanoProjector(view_set("ring8_poles"), (64, 128), (32, 32)).coverage == 1.0
    assert PanoProjector(view_set("ring8"), (64, 128), (32, 32)).coverage < 1.0


def test_roundtrip_erp_views_erp():
    H, W = 128, 256
    erp = smooth_sphere_field(H, W)
    proj = PanoProjector(view_set("dense18"), (H, W), (96, 96))
    rec = proj.to_erp(proj.to_views(erp))
    err = (rec - erp).abs().mean().item()
    assert err < 0.02, err


def test_seam_wrap():
    # a view centered on the seam (yaw 180) must sample both ERP edges smoothly
    H, W = 128, 256
    erp = smooth_sphere_field(H, W)
    proj = PanoProjector([View(180, 0, 90)], (H, W), (64, 64))
    v = proj.to_views(erp)[0]
    d = perspective_directions(View(180, 0, 90), 64, 64)
    x, y, z = d.unbind(-1)
    gt = torch.stack([torch.sin(3 * x) + y, torch.cos(2 * z) * x, y * y - z]).float()
    assert (v - gt).abs().mean().item() < 0.02


def test_warped_noise_is_white_and_shared():
    views = view_set("dense18")
    wn = WarpedNoise(views, (32, 32), (256, 512))  # latent-like sizes: 32x32 views, fine grid 8x a 32x64 ERP
    assert wn.min_cells >= 1
    eps = wn.sample(16, torch.Generator().manual_seed(0))  # (18, 16, 32, 32)
    assert abs(eps.mean().item()) < 0.02 and abs(eps.std().item() - 1) < 0.02
    # i.i.d. within a view: neighbouring pixels uncorrelated (nearest projection would give ~1)
    c = (eps[..., :, 1:] * eps[..., :, :-1]).mean().item()
    assert abs(c) < 0.03, c
    # shared across views: the same direction seen by two overlapping views gets the same noise
    proj = PanoProjector(views[:2], (256, 512), (32, 32))  # yaw 0 and 45 overlap by 45 deg
    d0 = perspective_directions(views[0], 32, 32)
    from sfe.geometry import rotation
    dc = d0 @ rotation(views[1].yaw, views[1].pitch)  # view-0 pixel directions in view-1 camera
    xn, yn = dc[..., 0] / dc[..., 2], -dc[..., 1] / dc[..., 2]
    m = (dc[..., 2] > 0) & (xn.abs() < 0.9) & (yn.abs() < 0.9)
    col = ((xn + 1) / 2 * 32).long().clamp(0, 31)
    row = ((yn + 1) / 2 * 32).long().clamp(0, 31)
    a, b = eps[0][:, m], eps[1][:, row[m], col[m]]
    r = ((a * b).mean() / (a.std() * b.std())).item()
    assert r > 0.5, r  # partial footprint overlap -> strongly but not perfectly correlated
    del proj


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
