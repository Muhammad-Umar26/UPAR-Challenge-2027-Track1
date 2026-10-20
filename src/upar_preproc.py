"""Shared preprocessing. The cache builder (training) and run.py (inference) MUST use this same function."""
import numpy as np
from PIL import Image

PAD_RGB = (124, 116, 104)   # ImageNet mean in uint8 -> ~0 after normalisation

def letterbox(img, H, W, mode="pad", pad_rgb=PAD_RGB):
    """PIL image -> uint8 array (H, W, 3).

    mode="pad":     keep aspect ratio, fit inside HxW, centre on a mean-coloured canvas.
    mode="stretch": plain resize to HxW (distorts aspect).
    """
    img = img.convert("RGB")
    w, h = img.size
    if mode == "stretch":
        return np.asarray(img.resize((W, H), Image.BICUBIC), dtype=np.uint8)
    s = min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    if (nw, nh) == (W, H):
        return np.asarray(img, dtype=np.uint8)
    canvas = Image.new("RGB", (W, H), pad_rgb)
    canvas.paste(img, ((W - nw) // 2, (H - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)

def load_resized(path, H, W, mode="pad"):
    with Image.open(path) as im:
        return letterbox(im, H, W, mode)