"""Saved full-start maps of the fixed p24 refinement; no target work."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path('/data2/neural-shrimp-data/explore/clark-rnn/n6-p24-refinement-001')
PIN='6f13a9ac62468a9fd43ab84cd0ebee0eeab8d45ceabbd54d444fa9b142ee1e89'

def main(out):
 t=time.monotonic();out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(ROOT/'manifest.json')==PIN
 inv=json.loads((ROOT/'manifest.json').read_text())['files']
 for name in ['run/batch.npz','run/inputs.npz']:
  p=ROOT/name;assert sha(p)==inv[name]['sha256']and p.stat().st_size==inv[name]['bytes']
 with np.load(ROOT/'run/inputs.npz')as z:g=z['gains'];d=z['deltas']
 with np.load(ROOT/'run/batch.npz')as z:period=z['period_proposals'];growth=z['lambda_per_step']
 fig,ax=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
 cmap=plt.get_cmap('turbo').copy();cmap.set_bad('#dedede')
 for i in range(2):
  p=np.ma.masked_equal(period[i::2].reshape(51,51).T,0);im=ax[0,i].pcolormesh(g,d,p,cmap=cmap,vmin=18,vmax=60,shading='nearest',rasterized=True)
  q=ax[1,i].pcolormesh(g,d,growth[i::2].reshape(51,51).T,cmap='RdBu_r',vmin=-.03,vmax=.03,shading='nearest',rasterized=True)
  ax[0,i].set_title(f'Native start {i}: tail recurrence period');ax[1,i].set_title(f'Native start {i}: finite mean tangent growth')
  ax[1,i].set_xlabel('Gain g');ax[0,i].set_xscale('log')
 for a in ax[:,0]:a.set_ylabel('Discrete leak / Euler parameter δ')
 fig.colorbar(im,ax=ax[0,:],label='Proposed period; gray = unresolved',fraction=.025,pad=.025)
 fig.colorbar(q,ax=ax[1,:],label='Growth per step (clipped color scale)',fraction=.025,pad=.025)
 fig.suptitle('Fixed iid matrix N=6, seed28184: all5,202 trajectories retained',y=.99)
 fig.text(.1,.015,'503-cell leading p24 component reaches crop edges;64/64 selected native cycles qualified.\nRecurrence/growth maps are numerical finite-horizon context, not shrimp/chaos or ODE proof.',fontsize=9)
 fig.subplots_adjust(left=.1,right=.82,top=.92,bottom=.13,hspace=.3,wspace=.13)
 for suffix in ['png','pdf']:fig.savefig(out/f'maps.{suffix}',dpi=160)
 plt.close(fig)
 m={'input_outer':PIN,'elapsed_seconds':time.monotonic()-t,'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}for p in out.iterdir()}};assert m['elapsed_seconds']<10 and sum(v['bytes']for v in m['files'].values())<2*1024**2
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps(m))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
