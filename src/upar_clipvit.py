"""Plain-PyTorch CLIP ViT image towers (same parameter names as open_clip) + a 40-way PAR head.

Inference needs only torch. open_clip is used only at training time, to fetch the pretrained weights.
The model takes the normalised 192x96 crops and resizes them to its own input size
(224x112 for ViT-B/16, 196x98 for ViT-L/14 by default).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

SPECS = {
    'clip_b16': dict(model='ViT-B-16', patch=16, width=768, layers=12, heads=12, out_dim=512, in_size=(224, 112)),
    'clip_l14': dict(model='ViT-L-14', patch=14, width=1024, layers=24, heads=16, out_dim=768, in_size=(196, 98)),
}
DIM_KEYS = ('patch', 'width', 'layers', 'heads', 'out_dim')


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
    def __init__(self, in_size=(224, 112), act='quickgelu', n_out=40, patch=16, width=768, layers=12, heads=12,
                 out_dim=512, arch='clip_b16'):
        super().__init__()
        self.in_size = tuple(in_size)
        self.visual = ClipViT(grid=(self.in_size[0] // patch, self.in_size[1] // patch), patch=patch, width=width,
                              layers=layers, heads=heads, out_dim=out_dim, act=act)
        self.head = nn.Linear(out_dim, n_out)
        self.export_cfg = {'arch': arch, 'in_size': list(self.in_size), 'act': act, 'mean': list(CLIP_MEAN),
                           'std': list(CLIP_STD), 'patch': patch, 'width': width, 'layers': layers, 'heads': heads,
                           'out_dim': out_dim}

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


def load_pretrained_visual(model='ViT-B-16', pretrained='openai'):
    import open_clip
    err = None
    for name in (model + '-quickgelu', model):
        try:
            vis = open_clip.create_model(name, pretrained=pretrained).visual.eval()
            act = 'quickgelu' if 'QuickGELU' in type(vis.transformer.resblocks[0].mlp.gelu).__name__ else 'gelu'
            return vis, act
        except Exception as e:
            err = e
    raise RuntimeError(f'could not load {model} ({pretrained}) weights: {err!r}')


def load_openai_visual():                      # kept for older code
    return load_pretrained_visual('ViT-B-16', 'openai')


def build_clip_par(pretrained=True, spec='clip_b16', in_size=None, act='quickgelu', tag='openai'):
    sp = SPECS[spec]
    in_size = tuple(in_size) if in_size else sp['in_size']
    assert all(v % sp['patch'] == 0 for v in in_size), f'{in_size} must be a multiple of the patch size {sp["patch"]}'
    dims = {k: sp[k] for k in DIM_KEYS}
    if not pretrained:
        return ClipPAR(in_size, act, 40, arch=spec, **dims)
    vis, act = load_pretrained_visual(sp['model'], tag)
    model = ClipPAR(in_size, act, 40, arch=spec, **dims)
    sd = {k: v.detach().float() for k, v in vis.state_dict().items()}
    del vis
    sd['positional_embedding'] = resize_pos_embed(sd['positional_embedding'], model.visual.grid)
    missing, unexpected = model.visual.load_state_dict(sd, strict=False)
    assert not missing, f'CLIP weights are missing keys: {missing}'
    if unexpected:
        print('ignored CLIP keys:', unexpected)
    return model


@torch.no_grad()
def check_against_open_clip(spec='clip_b16', device='cuda', tag='openai'):
    sp = SPECS[spec]
    vis, act = load_pretrained_visual(sp['model'], tag)
    g = 224 // sp['patch']
    ours = ClipViT(grid=(g, g), act=act, **{k: sp[k] for k in DIM_KEYS})
    missing, unexpected = ours.load_state_dict({k: v.float() for k, v in vis.state_dict().items()}, strict=False)
    assert not missing, missing
    vis, ours = vis.float().to(device).eval(), ours.to(device).eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    a = vis(x)
    a = a[0] if isinstance(a, tuple) else a
    d = (a - ours(x)).abs().max().item()
    print(f'{spec}: activation={act} | max |open_clip - ours| = {d:.2e} | ignored keys: {unexpected}')
    assert d < 1e-3, 'ARCHITECTURE MISMATCH: do not train, send this output'
    del vis, ours
    torch.cuda.empty_cache()