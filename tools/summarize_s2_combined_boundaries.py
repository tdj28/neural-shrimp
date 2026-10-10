"""Fixed saved-only boundary overlay; no new model or derivative evaluations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves')
PINS={'attempt-003-target':'1a1b58e678f11db81ca7adb2821ca2ff84056768734073a11e70e004813a71f9','attempt-004-fold-plus':'231b1d448fd06b09a180861d4e67569d3ef0f42967557ba4e24dd7b4780284f6','attempt-005-ray51-flip':'b3ccbdafc68c8ce6981d286d7f7a4145624a9b8d6806e721471a06186b7135c9','attempt-006-ray32-fold':'5a407527b29bf0a94540e581b23967c5e956b175e2f6ecf1d7b51806e7688f7d','attempt-007-ray13-flip':'c889339446be01c3110303e311b06266e5a10b1bdacb1dd45f4ee06abd2e49a7'}
MAP=Path('/data2/neural-shrimp-data/explore/s2-28011-w587-512/map.npz')
MAP_PIN='5d9fc07f06bbc36025d2ec0b315be1b4cd812141eb0a0a4f9d5464d6b71fa874'
def ent(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 resource.setrlimit(resource.RLIMIT_AS,(1024*1024**2,1024*1024**2));start=time.monotonic()
 out=BASE/'combined-summary002';out.mkdir(exist_ok=False);records=[];rows=[]
 for name,pin in PINS.items():
  base=BASE/name;assert ent(base/'manifest.json')['sha256']==pin
  inv=read(base/'manifest.json')['files'];assert set(inv)=={str(p.relative_to(base))for p in base.rglob('*')if p.is_file()}-{'manifest.json'}
  for f,e in inv.items():assert ent(base/f)==e and not (base/f).stat().st_mode&0o222
  launch=read(base/'launcher-result.json');assert launch['exit_code']==0 and not launch['owned_group_live_pids']
  for pid in (launch['launcher_pid'],launch['owned_group_pid']):assert not Path('/proc',str(pid)).exists()
  outcome=read(base/'run/outcome.json');assert outcome['status']=='numerical_boundary_panel_recorded'
  sigma=read(base/'run/inputs.json')['sigma'];nodes=[]
  for f in sorted((base/'run').glob('node-*.json')):
   a=read(f);q=a['qualification'];assert q['qualified'];point=np.array(a['point'],dtype=float)
   ev=np.array(q['multipliers_real'])+1j*np.array(q['multipliers_imag']);critical=q['critical_index'];other=np.delete(ev,critical)
   row={'run':name,'file':f.name,'sigma':sigma,'sign':a['sign'],'index':a['index'],'hidden':float(point[13]),'output':float(point[14]),'critical_gap':float(min(abs(other-sigma))),'nearest_other_real':float(other[np.argmin(abs(other-sigma))].real),'nearest_other_imag':float(other[np.argmin(abs(other-sigma))].imag),'complement_radius':q['complement_radius'],'closure':q['closure'],'proper_divisor_separations':q['proper_divisor_separations']}
   rows.append(row);nodes.append(row)
  assert len(nodes)==sum(d['accepted']for d in outcome['directions'])==64
  records.append({'root':str(base),'manifest_sha256':pin,'outcome':outcome,'launch':launch,'nodes':len(nodes),'bytes':sum(e['bytes']for e in inv.values())})
 assert ent(MAP)['sha256']==MAP_PIN
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 m=np.load(MAP);lh,lo=m['log10_hidden'],m['log10_output'];status,period=m['status'],m['period'];layer=np.zeros(status.shape);layer[(status==2)&(period!=18)]=1;layer[(status==2)&(period==36)]=2;layer[(status==2)&(period==18)]=3
 axs[0].imshow(layer,origin='lower',extent=[lo[0],lo[-1],lh[0],lh[-1]],aspect='auto',interpolation='nearest',cmap=ListedColormap(['white','#ddd','#aaa','#555']),vmin=-.5,vmax=3.5)
 for name in PINS:
  for sign in [-1,1]:
   part=sorted([r for r in rows if r['run']==name and r['sign']==sign],key=lambda r:r['index'])
   if not part:continue
   c={'attempt-003-target':'#2377bf','attempt-004-fold-plus':'#2377bf','attempt-005-ray51-flip':'#c8552e','attempt-006-ray32-fold':'#20a076','attempt-007-ray13-flip':'#9d37ad'}[name];label=('fold' if part[0]['sigma']==1 else 'flip')+' '+name+' '+str(sign)
   x=np.log10([r['output']for r in part]);y=np.log10([r['hidden']for r in part]);axs[0].plot(x,y,'.-',color=c,lw=.8,markersize=2,label=label)
   unstable=np.array([r['complement_radius']>=1 for r in part]);axs[0].plot(x[unstable],y[unstable],'x',color='black',ms=4)
   axs[1].plot(range(len(part)),[r['complement_radius']for r in part],'.-',label=label,ms=2)
 axs[0].plot(np.log10(.8210596690584651),np.log10(71.337643833463),'r+',ms=8,label='certified interior center')
 axs[0].set(xlabel='log10 output learning rate',ylabel='log10 hidden learning rate',xlim=(-.105,-.07),ylim=(1.8,1.94),title='Carried numerical boundaries over historical period screen')
 axs[0].legend(fontsize=6);axs[1].axhline(1,color='black',lw=.7);axs[1].set(xlabel='Accepted node within segment',ylabel='Other-multiplier spectral radius',title='Crosses mark nonattracting complements on left');axs[1].legend(fontsize=6)
 fig.savefig(out/'combined-boundaries.png',dpi=160);fig.savefig(out/'combined-boundaries.pdf');plt.close(fig)
 result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source':ent(Path(__file__)),'runs':records,'map':{'path':str(MAP),**ent(MAP)},'rows':rows,'elapsed_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Numerical carried fold/flip nodes only; historical screen is context, no rigorous inter-node connection or shrimp proof.'}
 assert result['elapsed_seconds']<30 and result['peak_rss_kib']<512*1024
 save(out/'summary.json',result);(out/'source.py').write_bytes(Path(__file__).read_bytes());save(out/'manifest.json',{'files':{p.name:ent(p)for p in out.iterdir()}})
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps({'root':str(out),'manifest':ent(out/'manifest.json'),'rows':len(rows),'elapsed':result['elapsed_seconds']}))
if __name__=='__main__':main()
