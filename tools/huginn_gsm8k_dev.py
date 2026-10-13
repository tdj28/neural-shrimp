"""GSM8K development analysis (run H1b, as stated in its run card): accuracy per lam arm (strict, flexible) at r = 32,
paired McNemar against
native, tokens and stop behaviour."""
import json, sys
import numpy as np
sys.path.insert(0, '/data2/neural-shrimp/tools')
import huginn_arc_analysis as ha
f = sys.argv[1]
rows = [json.loads(l) for l in open(f)]
arms = sorted({r['arm'] for r in rows}, key=float)
by = {a: {r['index']: r for r in rows if r['arm'] == a} for a in arms}
common = sorted(set.intersection(*[set(v) for v in by.values()]))
print('problems with all arms:', len(common), 'rows', len(rows))
for metric in ('correct', 'correct_strict'):
    base = np.array([bool(by['1.0'][i][metric]) for i in common])
    print(f'== {metric}')
    for a in arms:
        x = np.array([bool(by[a][i][metric]) for i in common])
        p, b, c = ha.mcnemar_less(base, x)
        mean, boots = ha.paired_bootstrap(base, x)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        toks = np.array([by[a][i]['new_tokens'] for i in common])
        stopped = np.mean([('Q:' in by[a][i]['text'] or '\nQuestion:' in by[a][i]['text']) for i in common])
        has_answer = np.mean([by[a][i]['strict_pred'] is not None for i in common])
        print(f'  lam {a:>5}: acc {x.mean():.3f}  diff {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  McNemar(less) p {p:.3g} ({b} lost, {c} gained)'
              f'  tokens median {np.median(toks):.0f} mean {toks.mean():.0f}  cap {np.mean(toks >= 256):.2f}  stop {stopped:.2f}  strict-format {has_answer:.2f}')
