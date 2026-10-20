"""Resumable uint8 image cache: one .npy memmap per split, built in chunks."""
import json
import os
import time
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from upar_preproc import load_resized

CHUNK = 512
_G = {}

def _init(root, npy, paths, H, W, mode):
    _G.update(root=Path(root), npy=npy, paths=paths, H=H, W=W, mode=mode)

def _work(cid):
    g = _G
    lo, hi = cid * CHUNK, min((cid + 1) * CHUNK, len(g["paths"]))
    mm = np.load(g["npy"], mmap_mode="r+")
    bad = []
    for i in range(lo, hi):
        try:
            mm[i] = load_resized(g["root"] / g["paths"][i], g["H"], g["W"], g["mode"])
        except Exception as e:
            mm[i] = 0
            bad.append((i, g["paths"][i], repr(e)))
    mm.flush()
    return cid, hi - lo, bad

def _atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

def build_split(split, data_dir, cache_dir, H, W, mode="pad", workers=2):
    """Build (or resume) <cache_dir>/<split>_images.npy of shape (N, H, W, 3) uint8.

    Also writes <split>_labels.npy (N, 40 uint8), <split>_index.csv (row, image, source),
    attributes.json, and <split>_done.json (finished chunk ids, updated after every chunk).
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(data_dir / "annotations" / "task1" / split / "gt.csv")
    paths = gt.iloc[:, 0].tolist()
    attrs = list(gt.columns[1:])
    N = len(paths)
    assert len(attrs) == 40, f"expected 40 attribute columns, got {len(attrs)}"

    pd.DataFrame({"row": range(N), "image": paths,
                  "source": [p.split("/")[0] for p in paths]}).to_csv(cache_dir / f"{split}_index.csv", index=False)
    np.save(cache_dir / f"{split}_labels.npy", gt.iloc[:, 1:].to_numpy(dtype=np.uint8))
    (cache_dir / "attributes.json").write_text(json.dumps(attrs))

    npy, state_f, bad_f = (cache_dir / f"{split}_images.npy", cache_dir / f"{split}_done.json",
                           cache_dir / f"{split}_bad.txt")
    meta = dict(H=H, W=W, mode=mode, N=N, chunk=CHUNK)
    done = set()
    state = json.loads(state_f.read_text()) if state_f.exists() else None
    if npy.exists() and state and state["meta"] == meta:
        done = set(state["done"])
    else:
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(N, H, W, 3))
        del mm
        bad_f.write_text("")
        if state_f.exists():
            state_f.unlink()

    n_chunks = ceil(N / CHUNK)
    todo = [c for c in range(n_chunks) if c not in done]
    print(f"{split}: {N} images, {len(done)}/{n_chunks} chunks already done, {len(todo)} to build")
    if todo:
        t0 = time.time()
        total = sum(min(CHUNK, N - c * CHUNK) for c in todo)
        with Pool(workers, initializer=_init, initargs=(data_dir, str(npy), paths, H, W, mode)) as pool, \
                tqdm(total=total, unit="img", desc=split) as bar:
            for cid, n, bad in pool.imap_unordered(_work, todo):
                done.add(cid)
                bar.update(n)
                if bad:
                    with open(bad_f, "a") as f:
                        f.writelines("\t".join(map(str, b)) + "\n" for b in bad)
                _atomic_write(state_f, json.dumps({"meta": meta, "done": sorted(done)}))
        print(f"{split}: built in {(time.time() - t0) / 60:.1f} min ({total / (time.time() - t0):.0f} img/s)")
    n_bad = sum(1 for _ in open(bad_f)) if bad_f.exists() else 0
    return {"split": split, "N": N, "chunks_done": len(done), "n_chunks": n_chunks, "bad_images": n_bad}