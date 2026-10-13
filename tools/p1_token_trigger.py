"""Exploratory: operating points of a within-reply token trigger (running max of the per-token projection on the push
direction) on L8 replies: false-trigger rate on non-hostile replies, trigger rate and trigger token on hostile ones."""
import sys
import numpy as np
sys.path.insert(0, 'tools')
import p1_detection as pd
D = '/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/msm/'
for L in (12, 18):
    tp_all, y_all, onset_flag = [], [], []
    for arm in ('p1-conf040', 'p1-conf045'):
        s = np.load(D + arm + '/hostility.npz')['classifier'][:, :, 0]
        tp = np.load(D + arm + '/in-context.npz')[f'token_proj_{L}'].astype(float)
        y = s > 0.5
        on = pd.onsets(y)
        R, N, T = tp.shape
        first = np.zeros((R, N), bool)
        for r in range(R):
            if on[r] < N: first[r, on[r]] = True
        tp_all.append(tp.reshape(-1, T)); y_all.append(y.ravel()); onset_flag.append(first.ravel())
    tp = np.concatenate(tp_all); y = np.concatenate(y_all); first = np.concatenate(onset_flag)
    run = np.fmax.accumulate(np.nan_to_num(tp, nan=-np.inf), axis=1)
    neg = run[~y]
    for fpr in (0.02, 0.05, 0.10):
        theta = np.quantile(neg[:, -1], 1 - fpr)
        hit = run[first] > theta
        trig = hit.any(1)
        tok = np.argmax(hit, axis=1)[trig]
        hit_all = (run[y] > theta).any(1)
        print(f'layer {L} FPR {fpr:.2f}: theta {theta:.3f}; onset replies triggered {trig.mean():.2f} (median token {np.median(tok) if trig.any() else float("nan"):.0f}, '
              f'quartiles {np.percentile(tok,[25,75]) if trig.any() else None}); all hostile replies triggered {hit_all.mean():.2f}')
