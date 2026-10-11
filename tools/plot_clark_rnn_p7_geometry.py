#!/usr/bin/env python3
"""Saved-only p7 geometry and descriptive cycle matches. No model calls."""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn')
PINS={
 'p7-boundary-curves-001':'107003b5ddd108b636701a9cd7083fea9e06b2d4b0a4d9f6818a119611b0b098',
 'p7-lowgain-extension-001':'01297f353bdcc889bee8ffe497fa3cea28186836f45596b120ddc8d12a088d93',
 'p7-fold-flip-localization-001':'09b35dd46571a7b3eceb1ff41949f26bb5c5291eb4b5e20a7def5fede45e7f2d',
 'p7-flip-curves-001':'ca70240e31119dceddb3689d845a4ffd3576d97efb5fdf5da82e7c19ba940c54',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cycle_match(a,b):
    """Symmetry-aware descriptive distance, never an orbit-identity proof."""
    assert a.shape==b.shape and a.ndim==2
    scale=max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    return min((float(np.max(np.abs(a-sign*np.roll(b,k,axis=0)))/scale),sign,k)
               for sign in [-1,1] for k in range(len(a)))
def read_array(p,key):
    m=json.loads(p.read_text());q=p.with_name(m['array_file'])
    if sha(q)!=m['array_sha256']:raise ValueError('array binding')
    with np.load(q,allow_pickle=False) as f:return f[key].copy()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    start=time.monotonic()
    def check():
        if time.monotonic()-start>30:raise TimeoutError('30s saved-only cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024:raise MemoryError('512MiB')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('clean committed source required')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10 required')
    args.out.mkdir(parents=True,exist_ok=False)
    result={'status':'incomplete','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'pins':PINS}
    try:
        runs={}
        for name,pin in PINS.items():
            p=BASE/name
            if sha(p/'manifest.json')!=pin:raise ValueError('manifest pin')
            inv=json.loads((p/'manifest.json').read_text())['files']
            if set(inv)!={str(q.relative_to(p)) for q in p.rglob('*') if q.is_file() and q!=p/'manifest.json'}:raise ValueError('inventory')
            for f,d in inv.items():
                q=p/f
                if q.is_symlink() or sha(q)!=d['sha256'] or q.stat().st_size!=d['bytes']:raise ValueError('file identity')
            runs[name]=json.loads((p/'run/result.json').read_text());check()
        rootdir=BASE/'p7-fold-flip-localization-001/run'
        roots={}
        for p in rootdir.glob('*-fold_flip_result.json'):
            m=json.loads(p.read_text());branch='lower' if 'lower' in m['tag'] else 'upper'
            roots[branch]={'meta':m,'cycle':read_array(p,'cycle')}
        raw=BASE/'p7-flip-curves-001/run';families={}
        # The final same-tag family return is the accepted corrector iterate.
        for p in sorted(raw.glob('*-family.json')):
            m=json.loads(p.read_text());families[m['tag']]=p
        flips=runs['p7-flip-curves-001']['traces'];matches=[];series=[]
        for t in flips:
            ns=t['nodes'];cycles=[]
            for n in ns:
                tag=n['tag'] if not n['tag'].endswith(':seed') else f"p7extend-{t['branch']}:{130 if t['branch']=='lower' else 28}-seed"
                cycles.append(read_array(families[tag],'cycle'))
            entry={'branch':t['branch'],'direction':t['direction'],'root_matches':{}}
            distance_series=[]
            for branch,root in roots.items():
                ds=[cycle_match(root['cycle'],c) for c in cycles]
                par=[max(abs(n['gain']-root['meta']['gain'])/10,abs(n['delta']-root['meta']['delta'])/.1) for n in ns]
                eligible=range(10,len(ns)) if branch==t['branch'] else range(1,len(ns))
                bystate=min(eligible,key=lambda i:ds[i][0]);byparam=min(eligible,key=lambda i:par[i])
                entry['root_matches'][branch]={label:{'index':i,'tag':ns[i]['tag'],'cycle_error':ds[i][0],'odd_sign':ds[i][1],'phase_shift':ds[i][2],'scaled_parameter_error':par[i],'gain':ns[i]['gain'],'delta':ns[i]['delta']} for label,i in [('nearest_cycle',bystate),('nearest_parameter',byparam)]}
                if branch==t['branch']:distance_series=[x[0] for x in ds]
            matches.append(entry);series.append((t,distance_series));check()
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig,axes=plt.subplots(1,3,figsize=(15,5.2),constrained_layout=True)
        colors={'lower':'#1565a8','upper':'#109d89'}
        for name in ['p7-boundary-curves-001','p7-lowgain-extension-001']:
            for t in runs[name]['traces']:
                ns=t['nodes']
                for a,b in zip(ns[:-1],ns[1:]):
                    axes[0].plot([a['gain'],b['gain']],[a['delta'],b['delta']],color=colors[t['branch']],lw=1.4,ls='-' if a['attracting_complement'] and b['attracting_complement'] else '--')
        for k,(t,ds) in enumerate(series):
            ns=t['nodes'];color=['#c04523','#e09a25','#903d92','#535bc5'][k]
            axes[0].plot([n['gain'] for n in ns],[n['delta'] for n in ns],color=color,lw=.7)
            for a,b in zip(ns[:-1],ns[1:]):
                axes[1].plot([(a['gain']-1.802)*1e6,(b['gain']-1.802)*1e6],[(a['delta']-.7866)*1e6,(b['delta']-.7866)*1e6],color=color,lw=1.1,ls='-' if a['attracting_complement'] and b['attracting_complement'] else '--',alpha=.8)
            axes[2].plot(np.arange(len(ds))*.003,ds,color=color,label=f"{t['branch']} seed / direction {t['direction']:+d}")
        for b,r in roots.items():
            g=r['meta']['gain'];d=r['meta']['delta']
            axes[0].scatter(g,d,color=colors[b],marker='x',s=50,zorder=6)
            axes[1].scatter((g-1.802)*1e6,(d-.7866)*1e6,color=colors[b],marker='x',s=65,zorder=6)
        axes[0].set(xlim=(1.6,10),ylim=(.73,.86),xlabel='gain g',ylabel='leak / Euler step δ',title='Two sampled fold arms; finite endpoints')
        axes[1].set(xlabel='(g − 1.802) × 10⁶',ylabel='(δ − 0.7866) × 10⁶',title='All four flip traces overlap in projection')
        axes[2].set(xlabel='accumulated scaled arclength',ylabel='phase / odd-sign cycle distance',title='Distance back to each trace’s own seed')
        axes[2].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.15)
        fig.suptitle('Fixed random N4 RNN, period7: numerical geometry remains tongue-like\nSolid / dashed: attracting / nonattracting complement; ×: numerical fold–flip roots',fontsize=12)
        fig.savefig(args.out/'geometry.png',dpi=160);fig.savefig(args.out/'geometry.pdf');plt.close(fig)
        result.update(status='completed',cycle_matches=matches,scope='Saved-only descriptive analysis after seeing the flip outcome. Parameter proximity and nonzero nearest sampled cycle distance do not prove a common orbit, closed curve, cusp or complete shrimp. No new model/derivative/decomposition/solver call.',elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as e:result.update(status='failed',error=type(e).__name__+': '+str(e))
    finally:
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        inv={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in args.out.iterdir() if p.is_file()}
        if sum(d['bytes'] for d in inv.values())>8*1024**2:raise ValueError('8MiB cap')
        (args.out/'manifest.json').write_text(json.dumps({'files':inv},indent=2)+'\n')
        for p in args.out.iterdir():p.chmod(0o444)
        args.out.chmod(0o555)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cycle_matches']}))
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
