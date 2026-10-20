"""Resumable multi-label PAR training on the uint8 cache.

python upar_train.py --name NAME [--arch convnext_tiny|resnet50|clip_b16] [--epochs N] [--holdout PETA] [--final] [--speedtest]
Everything for a run lives in <runs-dir>/<name>/ on Drive: ckpt_last.pt (full state, resumed automatically),
best_state_dict.pt (plain weights of the best epoch), log.csv, val_logits_{best,last}.npy, meta.json.

Paths assume the Colab layout used in the project (/content/drive/MyDrive/upar/...).
"""
import argparse, csv, json, math, os, queue, random, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, '/content/drive/MyDrive/upar/code')
from upar_data import GPUAugment      # noqa: E402
from upar_eval import evaluate        # noqa: E402

SOURCES = ['Market1501', 'PA100k', 'PETA']
SRC_ID = {s: i for i, s in enumerate(SOURCES)}
CFG_KEYS = ['arch', 'epochs', 'batch_size', 'lr', 'balance', 'pw_cap', 'holdout', 'final', 'no_mask',
            'debug_steps', 'warmup_epochs', 'llrd', 'lr_head']

def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='run')
    p.add_argument('--cache-dir', default='/content/cache/h192w96_pad')
    p.add_argument('--runs-dir', default='/content/drive/MyDrive/upar/runs')
    p.add_argument('--arch', default='convnext_tiny', choices=['convnext_tiny', 'resnet50', 'clip_b16'])
    p.add_argument('--llrd', type=float, default=0.75, help='layer-wise lr decay (clip_b16 only)')
    p.add_argument('--lr-head', type=float, default=1e-3, help='lr of the new head (clip_b16 only)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--warmup-epochs', type=float, default=1.0)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--balance', type=float, default=0.0,
                   help='source share ~ n_s**balance: 0 = equal share per source, 1 = natural (unbalanced)')
    p.add_argument('--pw-cap', type=float, default=10.0, help='cap of the per-attribute pos_weight')
    p.add_argument('--holdout', default=None, choices=SOURCES, help='leave-one-source-out: train without this source')
    p.add_argument('--final', action='store_true', help='train on train+val, no validation')
    p.add_argument('--no-mask', action='store_true', help='ablation: do not mask never-annotated attributes')
    p.add_argument('--speedtest', action='store_true')
    p.add_argument('--gpu-hours', type=float, default=1.5)
    p.add_argument('--debug-steps', type=int, default=0, help='if >0: steps per epoch (smoke tests only)')
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args(argv)

# ----------------------------------------------------------------------------- data
def load_split(cache_dir, split):
    d = Path(cache_dir)
    images = np.load(d / f'{split}_images.npy', mmap_mode='r')
    labels = np.load(d / f'{split}_labels.npy')
    idx = pd.read_csv(d / f'{split}_index.csv')
    assert idx['source'].isin(SRC_ID).all() and len(idx) == len(images) == len(labels)
    return images, labels, idx['source'].map(SRC_ID).to_numpy(np.int64), idx['source'].to_numpy()

class Parts:
    """Logical concatenation of (images_memmap, labels, src) triples."""

    def __init__(self, triples):
        self.triples = triples
        self.offsets = np.cumsum([0] + [len(t[0]) for t in triples])
        self.labels = np.concatenate([t[1] for t in triples])
        self.src = np.concatenate([t[2] for t in triples])

    def __len__(self):
        return int(self.offsets[-1])

    def gather(self, ids):
        ids = np.sort(np.asarray(ids))
        xs = []
        for k, (im, _, _) in enumerate(self.triples):
            lo, hi = self.offsets[k], self.offsets[k + 1]
            sel = ids[(ids >= lo) & (ids < hi)] - lo
            if len(sel):
                xs.append(im[sel])
        x = xs[0] if len(xs) == 1 else np.concatenate(xs)
        return x, self.labels[ids], self.src[ids]

class Loader:
    """Background thread: batches of uint8 images + labels + source ids in pinned memory -> GPU."""

    def __init__(self, parts, order, batch, device='cuda', drop_last=True, depth=4):
        self.parts, self.order, self.B = parts, np.asarray(order), batch
        self.device, self.drop_last, self.depth = device, drop_last, depth

    def __len__(self):
        n = len(self.order)
        return n // self.B if self.drop_last else math.ceil(n / self.B)

    @staticmethod
    def _put(q, item, stop):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self, q, stop):
        try:
            for b in range(len(self)):
                if stop.is_set():
                    return
                x, y, s = self.parts.gather(self.order[b * self.B:(b + 1) * self.B])
                item = (torch.from_numpy(np.ascontiguousarray(x)).pin_memory(),
                        torch.from_numpy(np.ascontiguousarray(y)).pin_memory(),
                        torch.from_numpy(np.ascontiguousarray(s)).pin_memory())
                self._put(q, item, stop)
            self._put(q, None, stop)
        except Exception as e:
            self._put(q, e, stop)

    def __iter__(self):
        q, stop = queue.Queue(maxsize=self.depth), threading.Event()
        threading.Thread(target=self._produce, args=(q, stop), daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                x, y, s = item
                yield (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).float(),
                       s.to(self.device, non_blocking=True))
        finally:
            stop.set()

def sampling(src, row_ok, balance):
    """Rows to draw from and their probabilities: share of source s ~ n_s ** balance."""
    rows = np.where(row_ok)[0]
    s = src[rows]
    n_s = np.bincount(s, minlength=3).astype(np.float64)
    w = np.zeros(len(rows))
    for k in range(3):
        if n_s[k] > 0:
            w[s == k] = n_s[k] ** (balance - 1.0)
    return rows, w / w.sum()

def source_masks(labels, src, use_mask):
    """(3, 40) float: 0 where the source never has a positive (never annotated), computed on TRAIN rows only."""
    M = np.ones((3, 40), np.float32)
    if use_mask:
        for k in range(3):
            sel = src == k
            if sel.any():
                M[k] = (labels[sel].sum(0) > 0).astype(np.float32)
    return M

# ----------------------------------------------------------------------------- model / optim
def build_model(arch):
    if arch == 'convnext_tiny':
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 40)
        head = m.classifier[2]
    elif arch == 'resnet50':
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 40)
        head = m.fc
    else:
        from upar_clipvit import build_clip_par
        m = build_clip_par(pretrained=True)
        head = m.head
    return m, head

def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith('.bias') or 'layer_scale' in n else decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': wd},
                              {'params': no_decay, 'weight_decay': 0.0}], lr=lr)

def make_optimizer_clip(model, lr, lr_head, wd, llrd):
    # layer-wise lr decay for the CLIP tower (deeper blocks get a larger lr); the new head gets lr_head
    L = len(model.visual.transformer.resblocks)
    groups = {}
    for n, p in model.named_parameters():
        if n.startswith('head.'):
            lid = L + 2
        elif n.startswith('visual.transformer.resblocks.'):
            lid = int(n.split('.')[3]) + 1
        elif n.startswith(('visual.ln_post', 'visual.proj')):
            lid = L + 1
        else:
            lid = 0
        no_wd = p.ndim <= 1 or n.endswith('.bias') or 'embedding' in n
        if (lid, no_wd) not in groups:
            g_lr = lr_head if lid == L + 2 else lr * llrd ** (L + 1 - lid)
            groups[(lid, no_wd)] = {'params': [], 'lr': g_lr, 'weight_decay': 0.0 if no_wd else wd}
        groups[(lid, no_wd)]['params'].append(p)
    return torch.optim.AdamW(list(groups.values()), lr=lr)

def make_sched(opt, total, warm, min_ratio):
    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

# ----------------------------------------------------------------------------- train / eval
def run_epoch(model, loader, aug, opt, sched, scaler, M_t, pw_t, clip, cb=None, log_every=100):
    model.train()
    tot = torch.zeros((), device='cuda')
    n = 0
    for i, (xb, yb, sb) in enumerate(loader):
        x = aug.train(xb)
        m = M_t[sb]
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
        l = F.binary_cross_entropy_with_logits(logits.float(), yb, pos_weight=pw_t, reduction='none')
        loss = (l * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.detach()
        n += 1
        if cb is not None:
            cb(i)
        if (i + 1) % log_every == 0:
            print(f'    step {i + 1}/{len(loader)}  loss {loss.item():.4f}  lr {opt.param_groups[0]["lr"]:.2e}', flush=True)
    return (tot / max(n, 1)).item(), n

@torch.no_grad()
def predict(model, parts, aug, bs=256, max_batches=None):
    model.eval()
    loader = Loader(parts, np.arange(len(parts)), bs, drop_last=False)
    out = []
    for i, (xb, _, _) in enumerate(loader):
        with torch.autocast('cuda', dtype=torch.float16):
            lg = model(aug.eval(xb))
        out.append(lg.float().cpu())
        if max_batches and i + 1 >= max_batches:
            break
    return torch.cat(out).numpy()

def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def get_rng(sampler_rng):
    return {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(), 'sampler': sampler_rng.bit_generator.state}

def set_rng(st, sampler_rng):
    random.setstate(st['py'])
    np.random.set_state(st['np'])
    torch.set_rng_state(st['torch'])
    torch.cuda.set_rng_state(st['cuda'])
    sampler_rng.bit_generator.state = st['sampler']

def append_log(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float('nan')

# ----------------------------------------------------------------------------- main
def main():
    args = get_args()
    assert torch.cuda.is_available(), 'no GPU'
    assert not (args.final and args.holdout), '--final and --holdout cannot be combined'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = Path(args.runs_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    names = json.loads((Path(args.cache_dir) / 'attributes.json').read_text())
    tr, va = load_split(args.cache_dir, 'train'), load_split(args.cache_dir, 'val')
    parts_val = Parts([va[:3]])
    parts_train = Parts([tr[:3], va[:3]]) if args.final else Parts([tr[:3]])

    row_ok = np.ones(len(parts_train), bool)
    if args.holdout:
        row_ok &= parts_train.src != SRC_ID[args.holdout]
    rows, w = sampling(parts_train.src, row_ok, args.balance)
    M = source_masks(tr[1], tr[2], not args.no_mask)

    y_rows = parts_train.labels[rows].astype(np.float64)
    m_rows = M[parts_train.src[rows]].astype(np.float64)
    pos = (w[:, None] * m_rows * y_rows).sum(0)
    neg = (w[:, None] * m_rows * (1 - y_rows)).sum(0)
    pw = np.clip(neg / np.maximum(pos, 1e-12), 1.0, args.pw_cap)
    p0 = np.clip(pos / np.maximum(pos + neg, 1e-12), 1e-3, 1 - 1e-3)

    B = args.batch_size
    steps_per_epoch = args.debug_steps if args.debug_steps > 0 else len(rows) // B
    total_steps = args.epochs * steps_per_epoch
    warm = int(args.warmup_epochs * steps_per_epoch)

    shares = {SOURCES[k]: round(float(w[parts_train.src[rows] == k].sum()), 3) for k in range(3)}
    print(f'run {args.name} | arch {args.arch} | train rows {len(rows)} | steps/epoch {steps_per_epoch} | '
          f'epochs {args.epochs} | holdout {args.holdout} | final {args.final}')
    print('sampling share per source:', shares)
    for k in range(3):
        masked = [names[a] for a in range(40) if M[k, a] == 0]
        if masked:
            print(f'  loss masked for {SOURCES[k]}:', masked)
    print('pos_weight: min %.1f max %.1f, %d attributes at the cap %.0f' %
          (pw.min(), pw.max(), int((pw >= args.pw_cap - 1e-9).sum()), args.pw_cap), flush=True)

    model, head = build_model(args.arch)
    with torch.no_grad():
        nn.init.normal_(head.weight, std=0.01)
        head.bias.copy_(torch.tensor(np.log(p0 / (1 - p0)), dtype=torch.float32))
    model = model.cuda().to(memory_format=torch.channels_last)
    opt = (make_optimizer_clip(model, args.lr, args.lr_head, args.wd, args.llrd) if args.arch == 'clip_b16'
           else make_optimizer(model, args.lr, args.wd))
    sched = make_sched(opt, total_steps, warm, args.min_lr / args.lr)
    scaler = torch.amp.GradScaler('cuda')
    aug = GPUAugment(device='cuda')
    if args.arch == 'clip_b16':                     # CLIP's own normalisation
        from upar_clipvit import CLIP_MEAN, CLIP_STD
        aug.mean = torch.tensor(CLIP_MEAN, device='cuda').view(1, 3, 1, 1)
        aug.std = torch.tensor(CLIP_STD, device='cuda').view(1, 3, 1, 1)
    M_t = torch.tensor(M, device='cuda')
    pw_t = torch.tensor(pw, dtype=torch.float32, device='cuda')
    sampler_rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ speed test
    if args.speedtest:
        n_steps = 200
        order = sampler_rng.choice(rows, size=n_steps * B, replace=True, p=w)
        marks = {}

        def cb(i):
            if i == 19 or i == n_steps - 1:
                torch.cuda.synchronize()
                marks[i] = time.time()
        torch.cuda.reset_peak_memory_stats()
        run_epoch(model, Loader(parts_train, order, B), aug, opt, sched, scaler, M_t, pw_t, args.grad_clip, cb, log_every=50)
        sec_step = (marks[n_steps - 1] - marks[19]) / (n_steps - 1 - 19)
        peak = torch.cuda.max_memory_allocated() / 1e9
        predict(model, parts_val, aug, max_batches=2)
        t = time.time(); predict(model, parts_val, aug, max_batches=15); torch.cuda.synchronize()
        eval_ips = 15 * 256 / (time.time() - t)
        t = time.time()
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, run_dir / '_speedtest.pt')
        save_s = time.time() - t
        (run_dir / '_speedtest.pt').unlink()
        train_s = steps_per_epoch * sec_step
        eval_s = 0.0 if args.final else len(parts_val) / eval_ips
        per_epoch = train_s + eval_s + save_s
        print('\n================ SPEED TEST ================')
        print(f'{sec_step * 1000:.0f} ms/step = {B / sec_step:.0f} img/s | peak GPU memory {peak:.1f} GB (limit 15) | RSS {rss_gb():.1f} GB')
        print(f'seconds per epoch: train {train_s:.0f} + validation {eval_s:.0f} + checkpoint {save_s:.0f} = {per_epoch:.0f} s ({per_epoch / 60:.1f} min)')
        print('epochs that fit in a GPU-hour budget (90% of the budget is used, 10% kept as reserve):')
        for h in sorted({1.0, 1.5, 2.0, 3.0, args.gpu_hours}):
            print(f'   {h:>4.1f} h -> {int(h * 3600 * 0.9 // per_epoch)} epochs')
        return

    # ------------------------------------------------------------------ resume
    ck_path, log_path = run_dir / 'ckpt_last.pt', run_dir / 'log.csv'
    cfg = {k: getattr(args, k) for k in CFG_KEYS}
    start_epoch, best = 0, {'score': -1.0, 'epoch': 0}
    if ck_path.exists() and not args.no_resume:
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        diff = {k: (ck['cfg'].get(k), cfg[k]) for k in CFG_KEYS if ck['cfg'].get(k) != cfg[k]}
        if diff:
            sys.exit(f'REFUSING to resume: settings differ from the checkpoint (old, new): {diff}\n'
                     f'Use a new --name, or --no-resume to start over in this folder.')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        start_epoch, best = ck['epoch'], ck['best']
        set_rng(ck['rng'], sampler_rng)
        if log_path.exists():
            df = pd.read_csv(log_path)
            df[df['epoch'] <= start_epoch].to_csv(log_path, index=False)
        print(f'RESUMED from {ck_path} after epoch {start_epoch}/{args.epochs} (best so far {best})', flush=True)
        del ck
    elif log_path.exists() and args.no_resume:
        log_path.unlink()
    if start_epoch >= args.epochs:
        print('run already finished; nothing to do'); return

    (run_dir / 'meta.json').write_text(json.dumps({
        'cfg': cfg, 'attribute_names': names, 'sources': SOURCES, 'loss_mask': M.tolist(), 'pos_weight': pw.tolist(),
        'sampling_share': shares, 'model_cfg': getattr(model, 'export_cfg', {}),
        'note': 'state_dict keys are ' + args.arch + ' with a 40-way head'}, indent=1))

    # ------------------------------------------------------------------ epochs
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        order = sampler_rng.choice(rows, size=steps_per_epoch * B, replace=True, p=w)
        loss, _ = run_epoch(model, Loader(parts_train, order, B), aug, opt, sched, scaler, M_t, pw_t, args.grad_clip)
        torch.cuda.synchronize()
        t_train = time.time() - t0

        row = {'epoch': epoch + 1, 'lr': opt.param_groups[0]['lr'], 'train_loss': loss, 'sec_train': t_train}
        improved = False
        if not args.final:
            t1 = time.time()
            logits = predict(model, parts_val, aug)
            mean, per = evaluate(1 / (1 + np.exp(-logits)), parts_val.labels, va[3])
            row['sec_eval'] = time.time() - t1
            for k in ('Challenge_Avg', 'mA', 'Instance_F1', 'Label_F1'):
                row[k] = mean[k]
            for s in SOURCES:
                for k in ('Challenge_Avg', 'mA', 'Instance_F1'):
                    row[f'{s}_{k}'] = per[s][k]
            score = per[args.holdout]['Challenge_Avg'] if args.holdout else mean['Challenge_Avg']
            row['select_score'] = score
            np.save(run_dir / 'val_logits_last.npy', logits.astype(np.float32))
            if score > best['score']:
                improved = True
                best = {'score': float(score), 'epoch': epoch + 1}
                atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / 'best_state_dict.pt')
                np.save(run_dir / 'val_logits_best.npy', logits.astype(np.float32))
        peak = torch.cuda.max_memory_allocated() / 1e9
        row.update({'best_score': best['score'], 'best_epoch': best['epoch'], 'peak_gpu_gb': peak, 'rss_gb': rss_gb()})
        append_log(log_path, row)

        t2 = time.time()
        atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best': best, 'cfg': cfg,
                     'rng': get_rng(sampler_rng)}, ck_path)
        msg = f'epoch {epoch + 1}/{args.epochs} | loss {loss:.4f} | train {t_train:.0f}s'
        if not args.final:
            msg += (f' | val {row["sec_eval"]:.0f}s | Challenge_Avg {mean["Challenge_Avg"]:.4f} mA {mean["mA"]:.4f} '
                    f'F1 {mean["Instance_F1"]:.4f} | per source ' +
                    ' '.join(f'{s[:3]} {per[s]["Challenge_Avg"]:.3f}' for s in SOURCES) +
                    f' | select {row["select_score"]:.4f}' + (' *best*' if improved else ''))
        print(msg + f' | ckpt {time.time() - t2:.0f}s | peak GPU {peak:.1f} GB', flush=True)
        if peak > 14.5:
            print('WARNING: GPU memory close to the 15 GB limit; lower --batch-size')

    atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / 'final_state_dict.pt')
    print(f'DONE. best epoch {best["epoch"]} (score {best["score"]:.4f}); weights: best_state_dict.pt / final_state_dict.pt')

if __name__ == '__main__':
    main()