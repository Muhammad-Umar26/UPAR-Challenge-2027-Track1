"""Re-implementation of the UPAR Track 1 metrics as described on the Codabench Evaluation page.
Assumption (not stated on the page): when a denominator is 0 the term is counted as 0.
auto_valid approximates the official "invalid GT" exclusion (the public gt.csv has no invalid marker).
Usage as a module: evaluate(probs, gt, sources, threshold=0.5) -> dict (mean over datasets)."""
import numpy as np

def _metrics(prob, gt, valid=None, thr=0.5):
    pred = prob > thr
    g = gt > 0.5
    if valid is None:
        valid = np.ones(gt.shape[1], bool)
    # label-based, only attributes valid in this dataset
    tp = (pred & g).sum(0); fn = (~pred & g).sum(0)
    tn = (~pred & ~g).sum(0); fp = (pred & ~g).sum(0)
    pos = tp / np.maximum(tp + fn, 1); neg = tn / np.maximum(tn + fp, 1)
    ma = ((pos + neg) / 2)[valid].mean()
    f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1))[valid].mean()
    # instance-based over valid attributes
    P = pred[:, valid]; G = g[:, valid]
    inter = (P & G).sum(1); union = (P | G).sum(1)
    acc = np.where(union > 0, inter / np.maximum(union, 1), 0).mean()
    prec_i = np.where(P.sum(1) > 0, inter / np.maximum(P.sum(1), 1), 0)
    rec_i = np.where(G.sum(1) > 0, inter / np.maximum(G.sum(1), 1), 0)
    prec, rec = prec_i.mean(), rec_i.mean()
    f1i = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    avg = 2 * ma * f1i / (ma + f1i) if ma + f1i > 0 else 0.0
    return dict(Challenge_Avg=avg, mA=ma, Label_F1=f1, Instance_Acc=acc,
                Instance_Prec=prec, Instance_Rec=rec, Instance_F1=f1i)

def evaluate(prob, gt, sources, thr=0.5, valid_by_source=None, auto_valid=True):
    """Each dataset is scored on its own, then the metrics are averaged over datasets."""
    prob, gt, sources = np.asarray(prob), np.asarray(gt), np.asarray(sources)
    per = {}
    for s in sorted(set(sources)):
        m = sources == s
        v = None if valid_by_source is None else valid_by_source.get(s)
        if v is None and auto_valid:
            v = gt[m].sum(0) > 0   # approximation: attributes never positive in a dataset ~ 'invalid' there
        per[s] = _metrics(prob[m], gt[m], v, thr)
    keys = next(iter(per.values())).keys()
    mean = {k: float(np.mean([per[s][k] for s in per])) for k in keys}
    return mean, per