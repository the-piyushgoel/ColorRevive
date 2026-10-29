"""Colour-space helpers, metrics and the inference routine shared by train/eval/app."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---- normalisation: L in [0,100] -> [-1,1];  a,b in [-110,110] -> [-1,1] ----

def rgb_to_lab_tensors(rgb_uint8):
    """HxWx3 uint8 -> (L: 1xHxW, ab: 2xHxW) float tensors in [-1,1]."""
    lab = rgb2lab(rgb_uint8).astype(np.float32)
    L = lab[..., :1] / 50.0 - 1.0
    ab = lab[..., 1:] / 110.0
    return (torch.from_numpy(L.transpose(2, 0, 1).copy()),
            torch.from_numpy(ab.transpose(2, 0, 1).copy()))


def lab_tensors_to_rgb(L, ab):
    """Batched tensors (Bx1xHxW, Bx2xHxW) in [-1,1] -> float RGB array BxHxWx3 in [0,1]."""
    L = (L.detach().cpu() + 1.0) * 50.0
    ab = ab.detach().cpu().clamp(-1, 1) * 110.0
    lab = torch.cat([L, ab], 1).permute(0, 2, 3, 1).numpy().astype(np.float64)
    return np.stack([lab2rgb(x) for x in lab]).astype(np.float32)


# ---- metrics ----

def psnr(a, b):
    return peak_signal_noise_ratio(a, b, data_range=1.0)


def ssim(a, b):
    return structural_similarity(a, b, channel_axis=-1, data_range=1.0)


def colorfulness(rgb):
    """Hasler & Süsstrunk (2003) colourfulness; higher = more vivid."""
    r, g, b = [rgb[..., i] * 255 for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


# ---- inference on arbitrary-size photographs ----

@torch.no_grad()
def colorize(G, img, device, size=128, max_side=1024, ab_scale=1.0):
    """Colourise a PIL image.

    The network sees a size x size luminance map; the predicted chroma is upsampled
    and merged with the ORIGINAL full-resolution luminance so fine detail is kept.
    Returns (grayscale_pil, colorized_pil, original_rgb_pil_resized).
    """
    img = img.convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    full = np.asarray(img)
    L_full = rgb2lab(full)[..., 0]

    small = np.asarray(img.resize((size, size), Image.BICUBIC))
    L_small = rgb2lab(small)[..., 0] / 50.0 - 1.0
    x = torch.from_numpy(L_small.astype(np.float32))[None, None].to(device)

    G.eval()
    ab = G(x)
    ab = F.interpolate(ab, size=L_full.shape, mode="bicubic", align_corners=False)
    ab = ab.clamp(-1, 1)[0].cpu().numpy().transpose(1, 2, 0) * 110.0 * ab_scale
    ab = np.clip(ab, -110, 110)

    color = lab2rgb(np.concatenate([L_full[..., None], ab], -1))
    gray = lab2rgb(np.concatenate([L_full[..., None], np.zeros_like(ab)], -1))
    to_pil = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
    return to_pil(gray), to_pil(color), img
