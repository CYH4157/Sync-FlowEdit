"""Thin wrappers around diffusers flow models (SD3, FLUX.1-dev).

Every backend exposes the same small interface used by the editing loops:
    encode(img)                -> latent  (B, C, h, w), already shifted/scaled
    decode(latent)             -> img in [-1, 1]
    text(prompt)               -> opaque conditioning object
    velocity(zs, conds, gs, s) -> list of velocity tensors (batched in one forward pass)
    sigmas(T, latent_hw)       -> descending noise levels, length T + 1, last = 0
Velocity convention (diffusers rectified flow): z_s = (1 - s) x0 + s eps, v = eps - x0,
so an Euler step from s to s_next is z + (s_next - s) * v.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Cond:
    """Text conditioning for a single prompt (batch dim = 1)."""
    embeds: torch.Tensor
    pooled: torch.Tensor
    neg_embeds: torch.Tensor | None = None  # SD3 only (CFG)
    neg_pooled: torch.Tensor | None = None
    text_ids: torch.Tensor | None = None  # FLUX only


class FlowBackend:
    vae_scale = 8
    max_batch = 8  # max latents per transformer call; set by caller for memory

    # ---------------- VAE ----------------
    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        lat = vae.encode(img.to(vae.device, vae.dtype)).latent_dist.mode()
        return (lat - vae.config.shift_factor) * vae.config.scaling_factor

    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        lat = lat.to(vae.device, vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
        return vae.decode(lat, return_dict=False)[0].clamp(-1, 1)

    @torch.no_grad()
    def encode_erp(self, img: torch.Tensor, pad_px: int = 64) -> torch.Tensor:
        """Encode an ERP image with circular horizontal padding (no seam from VAE zero padding)."""
        x = torch.cat([img[..., -pad_px:], img, img[..., :pad_px]], dim=-1)
        lat = self.encode(x)
        p = pad_px // self.vae_scale
        return lat[..., p:-p]

    @torch.no_grad()
    def decode_erp(self, lat: torch.Tensor, pad_lat: int = 8) -> torch.Tensor:
        z = torch.cat([lat[..., -pad_lat:], lat, lat[..., :pad_lat]], dim=-1)
        img = self.decode(z)
        p = pad_lat * self.vae_scale
        return img[..., p:-p]

    # ---------------- transformer ----------------
    def velocity(self, zs, conds, guidances, sigma):
        """zs: list of (b_i, C, h, w); all with the same h, w. Returns list of velocities."""
        sizes = [z.shape[0] for z in zs]
        z = torch.cat(zs)
        c = [cd for cd, n in zip(conds, sizes) for _ in range(n)]
        g = [gg for gg, n in zip(guidances, sizes) for _ in range(n)]
        outs = []
        for i in range(0, z.shape[0], self.max_batch):
            outs.append(self._velocity(z[i : i + self.max_batch], c[i : i + self.max_batch], g[i : i + self.max_batch], sigma))
        return list(torch.cat(outs).split(sizes))


class SD3Backend(FlowBackend):
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers", device="cuda",
                 dtype=torch.float16, drop_t5=False):
        from diffusers import StableDiffusion3Pipeline

        kw = dict(torch_dtype=dtype)
        if drop_t5:  # saves ~10 GB, slightly worse prompt following
            kw.update(text_encoder_3=None, tokenizer_3=None)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **kw).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device, self.dtype = torch.device(device), dtype

    @torch.no_grad()
    def text(self, prompt: str, negative: str = "") -> Cond:
        pe, npe, pp, npp = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, device=self.device,
            num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative,
        )
        return Cond(embeds=pe, pooled=pp, neg_embeds=npe, neg_pooled=npp)

    @torch.no_grad()
    def _velocity(self, z, conds, gs, sigma):
        b = z.shape[0]
        emb = torch.cat([c.neg_embeds for c in conds] + [c.embeds for c in conds])
        pooled = torch.cat([c.neg_pooled for c in conds] + [c.pooled for c in conds])
        t = torch.full((2 * b,), float(sigma) * 1000.0, device=self.device, dtype=self.dtype)
        out = self.pipe.transformer(
            hidden_states=torch.cat([z, z]).to(self.dtype), timestep=t,
            encoder_hidden_states=emb, pooled_projections=pooled, return_dict=False,
        )[0].float()
        v_unc, v_cond = out.chunk(2)
        g = torch.tensor(gs, device=z.device, dtype=torch.float32).view(-1, 1, 1, 1)
        return v_unc + g * (v_cond - v_unc)

    def sigmas(self, T, latent_hw):
        sch = self.pipe.scheduler
        sch.set_timesteps(T, device=self.device)
        return sch.sigmas.float().cpu().tolist()


class FluxBackend(FlowBackend):
    """FLUX.1-dev is guidance-distilled: guidance is an input embedding, no CFG pass."""

    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", device="cuda", dtype=torch.bfloat16):
        from diffusers import FluxPipeline

        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device, self.dtype = torch.device(device), dtype

    @torch.no_grad()
    def text(self, prompt: str, negative: str = "") -> Cond:
        pe, pp, ids = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=self.device, num_images_per_prompt=1, max_sequence_length=512,
        )
        return Cond(embeds=pe, pooled=pp, text_ids=ids)

    @staticmethod
    def pack(z):  # (B, C, h, w) -> (B, h/2 * w/2, 4C), 2x2 patchify as in FLUX
        b, c, h, w = z.shape
        return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)

    @staticmethod
    def unpack(x, h, w):
        b, _, d = x.shape
        c = d // 4
        return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    @staticmethod
    def image_ids(h, w, device, dtype):
        ids = torch.zeros(h // 2, w // 2, 3)
        ids[..., 1] = torch.arange(h // 2)[:, None]
        ids[..., 2] = torch.arange(w // 2)[None, :]
        return ids.reshape(-1, 3).to(device, dtype)

    @torch.no_grad()
    def _velocity(self, z, conds, gs, sigma):
        b, _, h, w = z.shape
        out = self.pipe.transformer(
            hidden_states=self.pack(z).to(self.dtype),
            timestep=torch.full((b,), float(sigma), device=self.device, dtype=self.dtype),
            guidance=torch.tensor(gs, device=self.device, dtype=torch.float32),
            pooled_projections=torch.cat([c.pooled for c in conds]),
            encoder_hidden_states=torch.cat([c.embeds for c in conds]),
            txt_ids=conds[0].text_ids,
            img_ids=self.image_ids(h, w, self.device, self.dtype),
            return_dict=False,
        )[0]
        return self.unpack(out.float(), h, w)

    def sigmas(self, T, latent_hw):
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        sch = self.pipe.scheduler
        seq = (latent_hw[0] // 2) * (latent_hw[1] // 2)
        mu = calculate_shift(seq, sch.config.get("base_image_seq_len", 256), sch.config.get("max_image_seq_len", 4096),
                             sch.config.get("base_shift", 0.5), sch.config.get("max_shift", 1.15))
        sch.set_timesteps(sigmas=np.linspace(1.0, 1.0 / T, T), mu=mu, device=self.device)
        return sch.sigmas.float().cpu().tolist()


def load_backend(name: str, device="cuda", **kw) -> FlowBackend:
    if name == "sd3":
        return SD3Backend(device=device, **kw)
    if name == "flux":
        return FluxBackend(device=device, **kw)
    raise ValueError(name)


# Recommended settings from the FlowEdit paper (Kulikov et al., ICCV 2025).
DEFAULTS = {
    "sd3": dict(T=50, n_max=33, n_min=0, n_avg=1, g_src=3.5, g_tar=13.5),
    "flux": dict(T=28, n_max=24, n_min=0, n_avg=1, g_src=1.5, g_tar=5.5),
}


def to_tensor(pil_img) -> torch.Tensor:
    """PIL -> (1, 3, H, W) in [-1, 1]."""
    a = torch.from_numpy(np.asarray(pil_img.convert("RGB"))).float() / 127.5 - 1.0
    return a.permute(2, 0, 1).unsqueeze(0)


def to_pil(img: torch.Tensor):
    from PIL import Image

    a = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def resize(img: torch.Tensor, hw) -> torch.Tensor:
    return F.interpolate(img, size=hw, mode="bicubic", align_corners=False).clamp(-1, 1)
