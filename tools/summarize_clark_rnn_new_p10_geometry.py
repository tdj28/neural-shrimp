#!/usr/bin/env python3
"""All accepted native p10 boundary nodes; descriptive saved arithmetic only."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn/new-p10-boundary-curves-001')
PIN='7b0dd0f7924850552da37bb8f7d81a62105f8b8beede79b876fd1e14c874f741'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    if sha(RAW/'manifest.json')!=PIN:raise ValueError('input pin')
    inv=json.loads((RAW/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p!=RAW/'manifest.json'}:raise ValueError('inventory')
    for n,d in inv.items():
        p=RAW/n
        if p.is_symlink() or p.stat().st_size!=d['bytes'] or sha(p)!=d['sha256']:raise ValueError('input bytes')
    r=json.loads((RAW/'run/result.json').read_text());rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.5));colors={'fold':'#2673a8','flip':'#c6452b'}
    for t in r['traces']:
        ns=t['nodes'];x=np.array([n['gain'] for n in ns]);y=np.array([n['delta'] for n in ns]);label=f"{t['branch']} {t['direction']:+d}";color=colors[t['branch']];style='-' if t['direction']==1 else '--'
        axes[0].plot(x,y,style,color=color,label=label,lw=1.5);axes[0].scatter(x[-1],y[-1],marker='x',s=45,color=color)
        axes[1].plot(x,[n['complement_radius'] for n in ns],style,color=color,label=label)
        axes[2].semilogy(x,np.maximum([n['antipodal_defect'] for n in ns],1e-16),style,color=color,label=label)
        nsaved=[]
        for n in [ns[0],ns[-1]]:
            tag=n['tag'];npth=[p for p in (RAW/'run').glob('*-node.json') if json.loads(p.read_text())['tag']==tag]
            epth=[p for p in (RAW/'run').glob('*-node_eigen.json') if json.loads(p.read_text())['tag']==tag]
            if len(npth)!=1 or len(epth)!=1:raise ValueError('exact node/eigen join')
            def arrays(p):
                m=json.loads(p.read_text());q=p.with_name(m['array_file'])
                if sha(q)!=m['array_sha256']:raise ValueError('array')
                with np.load(q) as f:return {k:f[k].copy() for k in f.files}
            nd=arrays(npth[0]);ed=arrays(epth[0]);h=nd['half_return'];v=ed['eigenvectors'];mode=[]
            for i in range(4):
                vi=v[:,i];hv=h@vi;lam=np.vdot(vi,hv)/np.vdot(vi,vi);res=np.max(np.abs(hv-lam*vi))/max(np.max(np.abs(vi)),1e-100)
                mode.append({'native_multiplier':[float(ed['eigenvalues'][i].real),float(ed['eigenvalues'][i].imag)],'half_action_rayleigh':[float(lam.real),float(lam.imag)],'half_action_residual':float(res)})
            nsaved.append({'tag':tag,'node_sha256':sha(npth[0]),'eigen_sha256':sha(epth[0]),'saved_native_modes_half_action':mode})
        rows.append({'branch':t['branch'],'direction':t['direction'],'stop':t['stop'],'node_count':len(ns),'endpoint_diagnostics':nsaved})
    axes[0].set(xlabel='gain g',ylabel='leak delta',title='Native p10 boundary traces');axes[0].legend(fontsize=8)
    axes[1].axhline(1,color='black',lw=.8);axes[1].set(xlabel='gain g',ylabel='other-mode radius',title='Complementary attraction')
    axes[2].set(xlabel='gain g',ylabel='normalized antipodal defect',title='Symmetric fold / nonsymmetric flip')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('N4 / matrix seed 28174 — numerical nodes only; lines are a visual guide',fontsize=12);fig.tight_layout();fig.savefig(a.out/'geometry.png',dpi=170);plt.close(fig)
    out={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'input_manifest_sha256':PIN,'rows':rows,'elapsed_seconds':time.monotonic()-start,'scope':'Saved numerical geometry and H action on already saved native eigenvectors; no model/derivative/eig/SVD/solve. Rayleigh values are NOT new half-map eigenvalue certificates, particularly on nonsymmetric cycles. Flip cubic criticality is untested. Lines do not certify arcs or common-orbit projected intersections.'}
    (a.out/'result.json').write_text(json.dumps(out,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()}},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555);print(json.dumps(out))
if __name__=='__main__':main()
