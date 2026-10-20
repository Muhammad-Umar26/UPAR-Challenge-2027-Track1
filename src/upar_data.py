"""Fast training data path: uint8 memmap cache -> pinned batches -> GPU augmentation (no PIL in the loop)."""
import math
import queue
import random
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class CacheSplit:
    def __init__(self, cache_dir, split):
        d = Path(cache_dir)
        self.images = np.load(d / f"{split}_images.npy", mmap_mode="r")   # N,H,W,3 uint8
        self.labels = np.load(d / f"{split}_labels.npy")                  # N,40 uint8
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

class Prefetcher:
    """Background thread builds shuffled uint8 batches (pinned memory); the main thread copies them to the GPU."""

    def __init__(self, split, batch_size=128, shuffle=True, drop_last=True, device="cuda", depth=4, seed=0):
        self.split, self.B, self.shuffle, self.drop_last = split, batch_size, shuffle, drop_last
        self.device, self.depth, self.seed, self.epoch = device, depth, seed, 0
        self.pin = str(device).startswith("cuda")

    def __len__(self):
        n = len(self.split)
        return n // self.B if self.drop_last else math.ceil(n / self.B)

    @staticmethod
    def _put(q, item, stop):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self, q, order, stop):
        try:
            for b in range(len(self)):
                if stop.is_set():
                    return
                ids = np.sort(order[b * self.B:(b + 1) * self.B])      # sorted ids = friendlier memmap reads
                x = torch.from_numpy(np.ascontiguousarray(self.split.images[ids]))
                y = torch.from_numpy(self.split.labels[ids])
                if self.pin:
                    x, y = x.pin_memory(), y.pin_memory()
                self._put(q, (x, y), stop)
            self._put(q, None, stop)
        except Exception as e:                                          # surface errors in the main thread
            self._put(q, e, stop)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.split)) if self.shuffle else np.arange(len(self.split))
        q, stop = queue.Queue(maxsize=self.depth), threading.Event()
        threading.Thread(target=self._produce, args=(q, order, stop), daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                x, y = item
                yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).float()
        finally:
            stop.set()

class GPUAugment:
    """uint8 (B,H,W,3) on GPU -> normalised float (B,3,H,W). All ops are batched torch ops.

    Deliberately NO hue shift, channel shuffle or random grayscale: 24 of the 40 attributes are colours.
    Horizontal flip is safe: no attribute is left/right specific.
    """

    def __init__(self, device="cuda", flip=0.5, affine=0.8, rot_deg=5.0, zoom=(0.9, 1.1), shift=0.05,
                 photo=0.8, bright=0.25, contrast=0.25, sat=0.25, lowres=0.2, erase=0.25, channels_last=True):
        self.device, self.flip, self.affine, self.rot, self.zoom, self.shift = device, flip, affine, rot_deg, zoom, shift
        self.photo, self.bright, self.contrast, self.sat = photo, bright, contrast, sat
        self.lowres, self.erase, self.channels_last = lowres, erase, channels_last
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def _u(self, B, lo, hi):
        return torch.empty(B, device=self.device).uniform_(lo, hi)

    def _gate(self, B, p):
        return (torch.rand(B, device=self.device) < p).view(B, 1, 1, 1)

    def _finish(self, x):
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x.contiguous()

    @torch.no_grad()
    def eval(self, xb):
        x = xb.permute(0, 3, 1, 2).float().div_(255)
        return self._finish((x - self.mean) / self.std)

    @torch.no_grad()
    def train(self, xb):
        B, H, W, _ = xb.shape
        dev = self.device
        x = xb.permute(0, 3, 1, 2).float().div_(255)

        # photometric: brightness, contrast, saturation (each sample on/off with prob `photo`)
        on = self._gate(B, self.photo).float()
        x = x * (1 + on * self._u(B, -self.bright, self.bright).view(B, 1, 1, 1))
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * (1 + on * self._u(B, -self.contrast, self.contrast).view(B, 1, 1, 1)) + m
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = ((x - g) * (1 + on * self._u(B, -self.sat, self.sat).view(B, 1, 1, 1)) + g).clamp_(0, 1)

        # low-resolution simulation (domain shift): down-up sample a random subset, one scale per batch
        if self.lowres > 0:
            s = random.uniform(0.4, 0.8)
            small = F.interpolate(x, scale_factor=s, mode="bilinear", antialias=True,
                                  align_corners=False, recompute_scale_factor=True)
            low = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
            x = torch.where(self._gate(B, self.lowres), low, x)

        x = (x - self.mean) / self.std

        if self.flip > 0:
            x = torch.where(self._gate(B, self.flip), x.flip(3), x)

        # small rotation / zoom / shift in one grid_sample; zeros after normalisation == mean colour == pad colour
        if self.affine > 0:
            a = (torch.rand(B, device=dev) < self.affine).float()
            ang = self._u(B, -self.rot, self.rot) * math.pi / 180 * a
            z = 1 + (self._u(B, self.zoom[0], self.zoom[1]) - 1) * a
            tx, ty = self._u(B, -self.shift, self.shift) * a, self._u(B, -self.shift, self.shift) * a
            c, s_ = torch.cos(ang), torch.sin(ang)
            theta = torch.zeros(B, 2, 3, device=dev)
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = c / z, -s_ * (H / W) / z, tx
            theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = s_ * (W / H) / z, c / z, ty
            grid = F.affine_grid(theta, list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        # random erasing (occlusion), filled with the mean colour (0 after normalisation)
        if self.erase > 0:
            area = self._u(B, 0.02, 0.15) * H * W
            r = torch.exp(self._u(B, math.log(0.3), math.log(3.3)))       # r = h / w
            eh, ew = torch.sqrt(area * r).clamp(max=H - 1), torch.sqrt(area / r).clamp(max=W - 1)
            top, left = torch.rand(B, device=dev) * (H - eh), torch.rand(B, device=dev) * (W - ew)
            ys = torch.arange(H, device=dev).view(1, H, 1)
            xs = torch.arange(W, device=dev).view(1, 1, W)
            box = ((ys >= top.view(B, 1, 1)) & (ys < (top + eh).view(B, 1, 1)) &
                   (xs >= left.view(B, 1, 1)) & (xs < (left + ew).view(B, 1, 1)))
            box = box & self._gate(B, self.erase).view(B, 1, 1)
            x = x.masked_fill(box.unsqueeze(1), 0.0)

        return self._finish(x)