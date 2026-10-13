"""Exploratory: how early within a reply does the pinned classifier, applied to the reply's prefix, flag a reply that
ends hostile? L8 replies (onset replies vs non-hostile at-risk replies), prefixes of k words."""
import json, sys
import numpy as np
sys.path.insert(0, 'tools')
import p1_detection as pd
import hostility_score as hs
D = '/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/msm/'
rng = np.random.default_rng(0)
pos, neg = [], []
for arm in ('p1-conf040', 'p1-conf045'):
    s = np.load(D + arm + '/hostility.npz')['classifier'][:, :, 0]
    texts = json.load(open(D + arm + '/sample.json'))['texts']
    on = pd.onsets(s > 0.5)
    R, N = s.shape
    for r in range(R):
        if on[r] < N: pos.append(texts[r][on[r]])
        for t in range(min(on[r], N)):
            neg.append(texts[r][t])
pos = [pos[i] for i in rng.choice(len(pos), min(600, len(pos)), replace=False)]
neg = [neg[i] for i in rng.choice(len(neg), 1200, replace=False)]
clf = hs.Classifier(8, 'cpu')
lens = [len(t.split()) for t in pos]
print('onset replies', len(pos), 'median words', np.median(lens))
for k in (3, 5, 8, 12, 20, 1000):
    cut = lambda t: ' '.join(t.split()[:k])
    sp = clf([cut(t) for t in pos])[:, 0]; sn = clf([cut(t) for t in neg])[:, 0]
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; sc = np.r_[sp, sn]
    th5 = np.quantile(sn, 0.95)
    print(f'first {k} words: AUC {pd.auc(y, sc):.3f}; flagged at 5% FPR {np.mean(sp > th5):.2f}; flagged at 0.5 {np.mean(sp > 0.5):.2f} (false {np.mean(sn > 0.5):.3f})')
