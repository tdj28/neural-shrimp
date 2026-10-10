#!/usr/bin/env python3
"""Saved-only combined numerical RNN boundary geometry; no model evaluation."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn')
PINS={'p5-boundary-curves-001':'8f56a841ca05e2f81d8ebbdcd1e812da03c8309af6568170dc27980352a24fd0','p5-lowgain-extension-001':'c3de676fd0ded289f9214ec1aba38eead7d1e166a15c33a692e4fe306c88aae9'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic();runs={}
    for name,pin in PINS.items():
        p=BASE/name
        if sha(p/'manifest.json')!=pin:raise ValueError('manifest')
        inv=json.loads((p/'manifest.json').read_text())['files']
        if set(inv)!={str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f!=p/'manifest.json'}:raise ValueError('inventory')
        for f,d in inv.items():
            q=p/f
            if q.is_symlink() or sha(q)!=d['sha256'] or q.stat().st_size!=d['bytes']:raise ValueError('bytes')
        runs[name]=json.loads((p/'run/result.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),constrained_layout=True);curves=[]
    for sign,color,label in [(1,'#1167aa','Period5 fold (+1)'),(-1,'#ce6718','Period5 flip (−1)')]:
        old=runs['p5-boundary-curves-001']['traces'];low=next(t['nodes'] for t in old if t['sign']==sign and t['direction']==-1);high=next(t['nodes'] for t in old if t['sign']==sign and t['direction']==1);extension=next(t['nodes'] for t in runs['p5-lowgain-extension-001']['traces'] if t['sign']==sign)
        combined=list(reversed(high[1:]))+low+extension[1:]
        curves.append({'sign':sign,'nodes':combined,'label':label})
        for ax in axes:
            for u,v in zip(combined[:-1],combined[1:]):
                stable=u['attracting_complement'] and v['attracting_complement'];ax.plot([u['gain'],v['gain']],[u['delta'],v['delta']],color=color,lw=1.7,ls='-' if stable else '--',alpha=1 if stable else .7)
            for q in [combined[0],combined[-1]]:ax.scatter(q['gain'],q['delta'],s=38,marker='s',facecolors='white',edgecolors=color,zorder=5)
            turns=[i for i in range(1,len(extension)) if extension[i]['gain_tangent']*extension[i-1]['gain_tangent']<0]
            for i in turns:
                q=extension[i];ax.scatter(q['gain'],q['delta'],color=color,s=35,marker='o',zorder=5)
            idx=108 if sign==1 else 87;u,v=extension[idx-1:idx+1];ax.plot([u['gain'],v['gain']],[u['delta'],v['delta']],color=color,lw=5,zorder=5)
    axes[0].set(xlim=(5.5,16.4),ylim=(.62,.98),title='Actual numerical curves and retained endpoints')
    axes[1].set(xlim=(5.63,6.12),ylim=(.735,.875),title='Low-g turns and secondary multiplier brackets')
    for ax in axes:ax.set_xlabel('gain g');ax.set_ylabel('leak / Euler step δ');ax.grid(alpha=.15)
    handles=[Line2D([0],[0],color='#1167aa',lw=2,label='Period5 fold (+1)'),Line2D([0],[0],color='#ce6718',lw=2,label='Period5 flip (−1)'),Line2D([0],[0],color='gray',lw=2,label='Solid: attracting complement'),Line2D([0],[0],color='gray',lw=2,ls='--',label='Dashed: nonattracting complement'),Line2D([0],[0],color='gray',lw=5,label='Thick: second ±1 crossing bracket')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Fixed random N4 RNN, period5: numerical boundary organization\nCircles mark projection turns, not cusps; squares are finite trace endpoints',fontsize=12)
    fig.savefig(a.out/'geometry.png',dpi=180);fig.savefig(a.out/'geometry.pdf');plt.close(fig)
    result={'schema':'clark-rnn-combined-geometry-v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'pins':PINS,'elapsed_seconds':time.monotonic()-start,'scope':'Saved numerical curves only. Projected closeness/crossing does not identify a common full-state orbit. Secondary multiplier brackets are not validated codimension-two roots. No complete shrimp/interval/ODE claim.','curves':curves}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');inv={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()};(a.out/'manifest.json').write_text(json.dumps({'files':inv},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555);print(json.dumps({'out':str(a.out),'elapsed_seconds':result['elapsed_seconds']}))
if __name__=='__main__':main()
