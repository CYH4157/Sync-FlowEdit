"""Exercise the editing loops on CPU with a toy backend (no diffusion weights needed)."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sfe.models import Cond, FlowBackend  # noqa: E402
from sfe.sync_flowedit import EditParams, SyncConfig, erp_flowedit, sync_flowedit  # noqa: E402


class ToyBackend(FlowBackend):
    """'VAE' = 8x average pooling to 3 channels; 'transformer' = analytic velocity toward a color."""

    vae_scale = 8

    def __init__(self):
        self.device = torch.device("cpu")

    def encode(self, img):
        return F.avg_pool2d(img.float(), 8)

    def decode(self, lat):
        return F.interpolate(lat.float(), scale_factor=8, mode="nearest")

    def text(self, prompt, negative=""):
        color = {"red": [1.0, -1, -1], "blue": [-1, -1, 1.0]}.get(prompt, [0.0, 0, 0])
        t = torch.tensor(color).view(1, 3, 1, 1)
        return Cond(embeds=t, pooled=t)

    def _velocity(self, z, conds, gs, sigma):
        # ideal rectified-flow velocity if the data were a constant image of the prompt color
        x0 = torch.cat([c.embeds for c in conds]).expand_as(z)
        return (z - x0) / max(sigma, 1e-3)

    def sigmas(self, T, latent_hw):
        return torch.linspace(1, 0, T + 1).tolist()


def toy_erp(H=64, W=128):
    img = torch.zeros(1, 3, H, W)
    img[:, 0] = torch.linspace(-1, 1, W)
    img[:, 2] = torch.linspace(-1, 1, H)[:, None]
    return img


P = EditParams(T=10, n_max=8, g_src=1, g_tar=1)


def test_identity_edit_keeps_source():
    be, img = ToyBackend(), toy_erp()
    cfg = SyncConfig(views="dense18", view_px=32)
    out, _ = sync_flowedit(be, img, "red", "red", P, cfg)
    # same prompt -> zero velocity difference -> output = blended re-projection of the source
    assert (out - img).abs().mean() < 0.15


def test_edit_moves_towards_target_and_is_seamless():
    be, img = ToyBackend(), toy_erp()
    for sync in ["dv", "none"]:
        cfg = SyncConfig(views="dense18", view_px=32, sync=sync)
        out, extra = sync_flowedit(be, img, "red", "blue", P, cfg)
        assert out.shape == img.shape
        assert out[:, 2].mean() > img[:, 2].mean() + 0.3, sync  # got bluer
    cfg = SyncConfig(views="dense18", view_px=32, output="erp", noise="pano", src_mode="project")
    out, _ = sync_flowedit(be, img, "red", "blue", P, cfg)
    assert out.shape == img.shape


def test_warped_noise_loop():
    be, img = ToyBackend(), toy_erp()
    cfg = SyncConfig(views="dense18", view_px=32, noise="warp", dv_rescale="global")
    out, _ = sync_flowedit(be, img, "red", "blue", P, cfg)
    assert out.shape == img.shape and out[:, 2].mean() > img[:, 2].mean() + 0.3


def test_rotating_layout():
    be, img = ToyBackend(), toy_erp()
    cfg = SyncConfig(views="cube6", fov=100, view_px=32, rotate=True)
    out, extra = sync_flowedit(be, img, "red", "blue", P, cfg)
    assert out.shape == img.shape
    assert out[:, 2].mean() > img[:, 2].mean() + 0.3  # edit applied
    assert len(extra["views_cfg"]) == 18  # final decode uses the dense layout
    same, _ = sync_flowedit(be, img, "red", "red", P, cfg)
    assert (same - img).abs().mean() < 0.15  # no edit when prompts match


def test_dv_rescale_modes():
    be, img = ToyBackend(), toy_erp()
    for mode in ["global", "local"]:
        out, _ = sync_flowedit(be, img, "red", "blue", P, SyncConfig(views="dense18", view_px=32, dv_rescale=mode))
        assert out.shape == img.shape and out[:, 2].mean() > img[:, 2].mean() + 0.3, mode
        same, _ = sync_flowedit(be, img, "red", "red", P, SyncConfig(views="dense18", view_px=32, dv_rescale=mode))
        assert (same - img).abs().mean() < 0.15, mode


def test_view_prompts():
    be, img = ToyBackend(), toy_erp()
    cfg = SyncConfig(views="dense18", view_px=32, view_prompts="hint")
    out, _ = sync_flowedit(be, img, "red", "blue", P, cfg)
    assert out.shape == img.shape
    same, _ = sync_flowedit(be, img, "red", "red", P, cfg)
    assert (same - img).abs().mean() < 0.15
    out, _ = sync_flowedit(be, img, "red", "blue", P, SyncConfig(views="dense18", view_px=32, view_prompts="ground"))
    assert out[:, 2].mean() > img[:, 2].mean() + 0.3  # rules leave sky-free prompts untouched


def test_ground_prompt_rules():
    from sfe.prompts import ground_prompt
    assert ground_prompt("a photo of a road at sunset with an orange sky") == "a photo of a road in warm orange evening light"
    assert ground_prompt("a photo of a yard at night under a starry sky") == "a photo of a yard at night"
    assert ground_prompt("a photo of a field under a cloudy sky, covered in snow in winter") == "a photo of a field, covered in snow in winter"
    assert ground_prompt("a photo of a cinema with red seats") == "a photo of a cinema with red seats"


def test_ground_prompt_neutral():
    from sfe.prompts import ground_prompt
    t = "a photo of a dirt road at sunset with an orange sky"
    assert ground_prompt(t) == "a photo of a dirt road in warm orange evening light"
    assert ground_prompt(t, "neutral") == "a photo of a dirt road in the evening"
    from sfe.sync_flowedit import _view_category
    from sfe.geometry import View
    cfg = SyncConfig(down_pitch=80)
    assert _view_category(View(0, -50), cfg) == "mid" and _view_category(View(0, -90), cfg) == "down"
    assert _view_category(View(0, -50), SyncConfig()) == "down"


def test_erp_baseline_runs():
    be, img = ToyBackend(), toy_erp()
    out = erp_flowedit(be, img, "red", "blue", P)
    assert out.shape == img.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
