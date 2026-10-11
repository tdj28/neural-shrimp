"""Saved-only complete fixed N6 p6 patch visualization; no scientific model calls."""
import hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn/n6-p6-refinement-001')
PIN='0971c84135fbc7840148ffec9be94e6b8e514025e62ce5b70c04bec0b9de9fbe'
def main():
 start=time.monotonic();out=RAW.parent/'n6-p6-spatial-001';out.mkdir(exist_ok=False)
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 assert hashlib.sha256((RAW/'manifest.json').read_bytes()).hexdigest()==PIN
 m=json.loads((RAW/'manifest.json').read_text())['files'];assert set(m)=={str(f.relative_to(RAW))for f in RAW.rglob('*')if f.is_file()and f!=RAW/'manifest.json'}
 for n,d in m.items():
  p=RAW/n;assert not p.is_symlink()and p.stat().st_size==d['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']
 with np.load(RAW/'run/batch.npz')as f: periods=f['period_proposals'].reshape(51,51,2);growth=f['lambda_per_step'].reshape(51,51,2)
 with np.load(RAW/'run/inputs.npz')as f:g,d=f['gains'],f['deltas']
 fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained');colors=['#ededed','#6495ed','#d62f34','#ef943a','#9b57a4','#888888'];cmap=ListedColormap(colors)
 for k in range(2):
  p=periods[:,:,k];code=np.full(p.shape,5);code[p==0]=0
  for j,v in enumerate((6,12,18,24),1):code[p==v]=j
  im=axes[k].pcolormesh(g,d,code.T,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,6.5),6),shading='nearest');axes[k].set(title=f'Original start {k+1}',xlabel='gain g',ylabel='leak δ',xscale='log')
 cb=fig.colorbar(im,ax=axes[:2],ticks=range(6),shrink=.82);cb.ax.set_yticklabels(['unresolved','p6','p12','p18','p24','other'])
 im=axes[2].pcolormesh(g,d,np.maximum(growth[:,:,0],growth[:,:,1]).T,cmap='coolwarm',vmin=-.05,vmax=.05,shading='nearest');axes[2].set(title='Maximum finite-horizon growth',xlabel='gain g',ylabel='leak δ',xscale='log');fig.colorbar(im,ax=axes[2],label='per discrete step')
 fig.suptitle('N6 / seed28181 fixed refined patch — recurrence proposals; no window or chaos proof');fig.savefig(out/'patch.png',dpi=160);plt.close(fig)
 r=dict(input_manifest=PIN,elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid(),scope='saved-only complete patch, no new target evaluation',files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in out.iterdir()});assert r['elapsed_seconds']<30 and r['peak_rss_kib']<512*1024 and sum(x['bytes']for x in r['files'].values())<8*1024**2
 (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in out.iterdir()}},indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps(r))
if __name__=='__main__':main()
