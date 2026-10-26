"""UPAR 2027 Track 1 submission: weighted ensemble of one or more models (CLIP ViT via the shipped upar_clipvit.py,
or torchvision), logits averaged, optional horizontal-flip TTA with a time guard.

Only torch, torchvision, numpy and Pillow are used. Nothing here looks at sample['image'] or at the directory of
sample['image_path']: the same models and thresholds are used for every image.
"""
import atexit
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
BATCH_SIZE = 64

USE_TTA = True             # horizontal-flip TTA; switched off automatically when throughput is too low
N_ASSUMED = 40000          # test-set size assumed by the time guard (the platform log showed 29,248 images)
TIME_BUDGET_S = 220.0      # of the 300 s limit; the rest is margin
N_THREADS = 12

PAD_RGB = (124, 116, 104)  # must equal upar_preproc.PAD_RGB used to build the training cache
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_S = {}


def _log(msg):
    print(f"[run.py] {msg}", flush=True)


def letterbox(img, H, W, mode="pad"):
    """Identical to upar_preproc.letterbox (training cache): keep aspect ratio, centre on a mean-coloured canvas."""
    img = img.convert("RGB")
    w, h = img.size
    if mode == "stretch":
        return np.asarray(img.resize((W, H), Image.BICUBIC), dtype=np.uint8)
    s = min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    if (nw, nh) == (W, H):
        return np.asarray(img, dtype=np.uint8)
    canvas = Image.new("RGB", (W, H), PAD_RGB)
    canvas.paste(img, ((W - nw) // 2, (H - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _build(mcfg, n):
    arch = mcfg["arch"]
    if arch == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
        return m
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n)
        return m
    if arch.startswith("clip"):
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from upar_clipvit import ClipPAR            # shipped next to run.py; plain torch, no open_clip
        dims = {k: mcfg[k] for k in ("patch", "width", "layers", "heads", "out_dim") if k in mcfg}
        return ClipPAR(tuple(mcfg["in_size"]), mcfg["act"], n, **dims)
    raise ValueError(arch)


def _dequant(sd):
    """int8 weights are stored as <name>::q (int8) plus <name>::s (one fp16 scale per row)."""
    out = {}
    for k, v in sd.items():
        if k.endswith("::s"):
            continue
        if k.endswith("::q"):
            base = k[:-3]
            s = sd[base + "::s"].float()
            out[base] = (v.float().reshape(v.shape[0], -1) * s.reshape(-1, 1)).reshape(v.shape)
        else:
            out[k] = v
    return out


def load_model():
    t0 = time.time()
    with open(os.path.join(ASSETS, "config.json")) as f:
        cfg = json.load(f)
    names = list(cfg["attribute_names"])
    n = len(names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = cfg.get("members") or [dict(cfg, file="model.pth")]
    models = []
    for mcfg in members:
        model = _build(mcfg, n)
        sd = torch.load(os.path.join(ASSETS, mcfg["file"]), map_location="cpu", weights_only=True)
        model.load_state_dict(_dequant(sd))
        del sd
        model = model.eval().to(dev)
        if dev == "cuda":
            model = model.to(memory_format=torch.channels_last)
        mean = torch.tensor(mcfg.get("mean", MEAN), device=dev).view(1, 3, 1, 1)
        std = torch.tensor(mcfg.get("std", STD), device=dev).view(1, 3, 1, 1)
        models.append((model, mean, std, float(mcfg.get("weight", 1.0))))
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True

    thr = np.full(n, 0.5)
    tpath = os.path.join(ASSETS, "thresholds.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            td = json.load(f)
        m = dict(zip(td["attribute_names"], td["thresholds"]))
        thr = np.array([m.get(nm, 0.5) for nm in names], dtype=np.float64)
    thr = np.clip(thr, 1e-6, 1 - 1e-6)
    with open(os.path.join(ASSETS, "prior.json")) as f:
        pr = json.load(f)
    pm = dict(zip(pr["attribute_names"], pr["prior"]))
    _S.update(
        models=models, dev=dev, names=names, pos={nm: i for i, nm in enumerate(names)}, cols={},
        H=int(cfg["H"]), W=int(cfg["W"]), mode=cfg.get("mode", "pad"),
        bias=torch.tensor(np.log(thr / (1 - thr)), dtype=torch.float32, device=dev),
        prior=np.array([pm.get(nm, 0.0) for nm in names], dtype=np.float64),
        pool=ThreadPoolExecutor(N_THREADS), tta=bool(USE_TTA), t_boot=t0, t_first=None, n_done=0, n_batches=0,
        n_fail=0, has_thr=os.path.exists(tpath))
    _log(f"{len(models)} model(s) {[mc['arch'] for mc in members]} on {dev}; thresholds file: {_S['has_thr']}; "
         f"TTA: {_S['tta']}; load {time.time() - t0:.1f}s")
    atexit.register(_summary)


def _summary():
    if _S.get("t_first") is None:
        return
    dt = time.time() - _S["t_first"]
    _log(f"DONE {_S['n_done']} images in {dt:.1f}s ({_S['n_done'] / max(dt, 1e-9):.0f} img/s), "
         f"unreadable: {_S['n_fail']}, TTA at end: {_S['tta']}")


def _load(path):
    try:
        with Image.open(path) as im:
            return letterbox(im, _S["H"], _S["W"], _S["mode"])
    except Exception as e:
        return e


@torch.no_grad()
def _forward(x_uint8):
    dev = _S["dev"]
    x0 = x_uint8.to(dev).permute(0, 3, 1, 2).float().div_(255)
    z, wsum = 0.0, 0.0
    with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        for model, mean, std, w in _S["models"]:
            x = (x0 - mean) / std
            if dev == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            zm = model(x).float()
            if _S["tta"]:
                zm = (zm + model(x.flip(3)).float()) / 2
            z = z + w * zm
            wsum += w
    z = z / wsum
    p = torch.sigmoid(z - _S["bias"]).clamp_(1e-6, 1 - 1e-6)       # p > 0.5  <=>  sigmoid(z) > t_k
    return p.cpu().numpy().astype(np.float64)


def _columns(names):
    key = tuple(names)
    if key not in _S["cols"]:
        _S["cols"][key] = [_S["pos"].get(n, -1) for n in names]
        if -1 in _S["cols"][key]:
            _log(f"unknown attribute names (answered with 0.0): {[n for n in names if n not in _S['pos']]}")
    return _S["cols"][key]


def _time_guard():
    now = time.time()
    if _S["n_batches"] % 10 or _S["n_done"] < 640:
        return
    rate = _S["n_done"] / max(now - _S["t_first"], 1e-9)
    projected = (now - _S["t_boot"]) + (N_ASSUMED - _S["n_done"]) / rate
    if _S["tta"] and projected > TIME_BUDGET_S:
        _S["tta"] = False
        _log(f"TTA disabled: {rate:.0f} img/s projects {projected:.0f}s for {N_ASSUMED} images (budget {TIME_BUDGET_S:.0f}s)")
    if _S["n_batches"] % 50 == 0:
        _log(f"{_S['n_done']} images, {rate:.0f} img/s, projected total {projected:.0f}s, TTA {_S['tta']}")


def predict_batch(samples):
    if "models" not in _S:
        load_model()
    if _S["t_first"] is None:
        _S["t_first"] = time.time()
    names = list(samples[0]["attribute_names"]) if samples else list(_S["names"])
    cols = _columns(names)
    paths = [s["image_path"] for s in samples]
    arrs = list(_S["pool"].map(_load, paths))

    P = np.tile(_S["prior"], (len(samples), 1))                    # fallback for unreadable images
    ok = [i for i, a in enumerate(arrs) if isinstance(a, np.ndarray)]
    for i, a in enumerate(arrs):
        if not isinstance(a, np.ndarray):
            _S["n_fail"] += 1
            if _S["n_fail"] <= 20:
                _log(f"unreadable image, using the training prior: {paths[i]} ({a!r})")
    if ok:
        try:
            P[ok] = _forward(torch.from_numpy(np.stack([arrs[i] for i in ok])))
        except Exception as e:                                      # never crash the whole run
            _log(f"inference failed for a batch, using the training prior: {e!r}")
    bad = ~np.isfinite(P).all(axis=1)
    if bad.any():
        _log(f"{int(bad.sum())} non-finite rows replaced by the training prior")
        P[bad] = _S["prior"]
    P = np.clip(np.nan_to_num(P, nan=0.5), 0.0, 1.0)

    out = np.zeros((len(samples), len(names)), dtype=np.float64)    # map by NAME to the requested order
    for j, c in enumerate(cols):
        if c >= 0:
            out[:, j] = P[:, c]
    _S["n_done"] += len(samples)
    _S["n_batches"] += 1
    _time_guard()
    return out.tolist()