"""Plain-PyTorch CLIP ViT-B/16 image tower (same parameter names as open_clip) + 40-way PAR head.

Inference needs only torch. open_clip is used only at training time, to fetch the pretrained OpenAI weights.
Input: normalised (B, 3, 192, 96); the model upsamples to 224x112 (a 14 x 7 patch grid).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResBlock(nn.Module):
    def __init__(self, d, heads, act):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d, 4 * d)),
                                              ('gelu', QuickGELU() if act == 'quickgelu' else nn.GELU()),
                                              ('c_proj', nn.Linear(4 * d, d))]))

    def forward(self, x):
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))

class Blocks(nn.Module):
    def __init__(self, d, layers, heads, act):
        super().__init__()
        self.resblocks = nn.ModuleList([ResBlock(d, heads, act) for _ in range(layers)])

class ClipViT(nn.Module):
    def __init__(self, grid=(14, 7), patch=16, width=768, layers=12, heads=12, out_dim=512, act='quickgelu'):
        super().__init__()
        self.grid = tuple(grid)
        self.conv1 = nn.Conv2d(3, width, patch, patch, bias=False)
        s = width ** -0.5
        self.class_embedding = nn.Parameter(s * torch.randn(width))
        self.positional_embedding = nn.Parameter(s * torch.randn(grid[0] * grid[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Blocks(width, layers, heads, act)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(s * torch.randn(width, out_dim))

    def forward(self, x):
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for blk in self.transformer.resblocks:
            x = blk(x)
        return self.ln_post(x[:, 0]) @ self.proj

class ClipPAR(nn.Module):
    def __init__(self, in_size=(224, 112), act='quickgelu', n_out=40):
        super().__init__()
        self.in_size = tuple(in_size)
        self.visual = ClipViT(grid=(self.in_size[0] // 16, self.in_size[1] // 16), act=act)
        self.head = nn.Linear(self.visual.proj.shape[1], n_out)
        self.export_cfg = {'arch': 'clip_b16', 'in_size': list(self.in_size), 'act': act,
                           'mean': list(CLIP_MEAN), 'std': list(CLIP_STD)}

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.in_size:
            x = F.interpolate(x, size=self.in_size, mode='bilinear', align_corners=False)
        return self.head(self.visual(x))

def resize_pos_embed(pe, grid):
    cls, g = pe[:1], pe[1:]
    n = int(round(g.shape[0] ** 0.5))
    g = g.reshape(1, n, n, -1).permute(0, 3, 1, 2).float()
    g = F.interpolate(g, size=tuple(grid), mode='bicubic', align_corners=False, antialias=True)
    return torch.cat([cls, g.permute(0, 2, 3, 1).reshape(grid[0] * grid[1], -1).to(pe.dtype)], 0)

def load_openai_visual():
    import open_clip
    err = None
    for name in ('ViT-B-16-quickgelu', 'ViT-B-16'):
        try:
            vis = open_clip.create_model(name, pretrained='openai').visual.eval()
            act = 'quickgelu' if 'QuickGELU' in type(vis.transformer.resblocks[0].mlp.gelu).__name__ else 'gelu'
            return vis, act
        except Exception as e:
            err = e
    raise RuntimeError(f'could not load CLIP ViT-B/16 OpenAI weights: {err!r}')

def build_clip_par(pretrained=True, in_size=(224, 112), act='quickgelu'):
    if not pretrained:
        return ClipPAR(in_size, act)
    vis, act = load_openai_visual()
    model = ClipPAR(in_size, act)
    sd = {k: v.detach().float() for k, v in vis.state_dict().items()}
    sd['positional_embedding'] = resize_pos_embed(sd['positional_embedding'], model.visual.grid)
    missing, unexpected = model.visual.load_state_dict(sd, strict=False)
    assert not missing, f'CLIP weights are missing keys: {missing}'
    if unexpected:
        print('ignored CLIP keys:', unexpected)
    return model

@torch.no_grad()
def check_against_open_clip(device='cuda'):
    vis, act = load_openai_visual()
    ours = ClipViT(grid=(14, 14), act=act)
    missing, unexpected = ours.load_state_dict({k: v.float() for k, v in vis.state_dict().items()}, strict=False)
    assert not missing, missing
    vis, ours = vis.float().to(device).eval(), ours.to(device).eval()
    x = torch.randn(4, 3, 224, 224, device=device)
    a = vis(x)
    a = a[0] if isinstance(a, tuple) else a
    d = (a - ours(x)).abs().max().item()
    print(f'activation={act} | max |open_clip - ours| = {d:.2e} | ignored keys: {unexpected}')
    assert d < 1e-3, 'ARCHITECTURE MISMATCH: do not train, send this output'