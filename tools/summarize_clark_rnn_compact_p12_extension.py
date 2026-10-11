#!/usr/bin/env python3
"""All accepted native p12 boundary nodes; descriptive saved arithmetic only."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
INPUTS=[(Path('/data2/neural-shrimp-data/explore/clark-rnn/compact-p12-boundary-curves-001'),'af1817405d90499cde73d88777a00fcf8076a2f80bc4e8ce50e23d94084f91dd'),(Path('/data2/neural-shrimp-data/explore/clark-rnn/compact-p12-boundary-extension-001'),'b5227f318d31ec5d940ff27307e799ae5a62fec70575eebe5734c2d8ab90136e')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    traces=[]
    for RAW,PIN in INPUTS:
        if sha(RAW/'manifest.json')!=PIN:raise ValueError('input pin')
        inv=json.loads((RAW/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p!=RAW/'manifest.json'}:raise ValueError('inventory')
        for n,d in inv.items():
            p=RAW/n
            if p.is_symlink() or p.stat().st_size!=d['bytes'] or sha(p)!=d['sha256']:raise ValueError('input bytes')
        traces.extend(json.loads((RAW/'run/result.json').read_text())['traces'])
    r={'traces':traces};rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.5));colors={'fold':'#2673a8','flip':'#c6452b'}
    for t in r['traces']:
        ns=t['nodes'];x=np.array([n['gain'] for n in ns]);y=np.array([n['delta'] for n in ns]);label=f"{t['branch']} {t['direction']:+d}";color=colors[t['branch']];style='-' if t['direction']==1 else '--'
        axes[0].plot(x,y,style,color=color,label=label,lw=1.5);axes[0].scatter(x[-1],y[-1],marker='x',s=45,color=color)
        axes[1].semilogy(x,[n['complement_radius'] for n in ns],style,color=color,label=label)
        axes[2].semilogy(x,np.maximum([n['odd_companion_min_phase_distance'] for n in ns],1e-16),style,color=color,label=label)
        rows.append({'branch':t['branch'],'direction':t['direction'],'stop':t['stop'],'node_count':len(ns),'first':ns[0],'last':ns[-1]})
    axes[0].set(xlabel='gain g',ylabel='leak delta',title='Native p12 boundary traces');axes[0].legend(fontsize=8)
    axes[1].axhline(1,color='black',lw=.8);axes[1].set(xlabel='gain g',ylabel='other-mode radius',title='Complementary attraction')
    axes[2].set(xlabel='gain g',ylabel='normalized odd-companion distance',title='Distinct odd companion under all phases')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('N4 / matrix seed 28172 — numerical nodes only; lines are a visual guide',fontsize=12);fig.tight_layout();fig.savefig(a.out/'geometry.png',dpi=170);plt.close(fig)
    out={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'input_manifests':{str(p):h for p,h in INPUTS},'rows':rows,'elapsed_seconds':time.monotonic()-start,'scope':'Saved numerical geometry only; no model/derivative/eig/SVD/solve. First sampled slice losses do not exclude narrower instability/re-entry pockets. Flip cubic criticality is untested. Lines do not certify arcs or common-orbit projected intersections.'}
    (a.out/'result.json').write_text(json.dumps(out,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()}},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555);print(json.dumps(out))
if __name__=='__main__':main()
