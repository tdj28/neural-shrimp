import json,pathlib,resource,signal,hashlib
resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(20,)*2);signal.alarm(30)
mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines())}
if mem['MemAvailable']<16*1024**3:raise SystemExit('Below runtime memory floor')
import numpy as np
P=pathlib.Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64');out={}
z=np.load(P/'dss/p6-body/grid.npz');v=z['multipliers'][...,0]+1j*z['multipliers'][...,1];ok=z['ok'];delta=v[...,0]*v[...,1];bad=ok&(abs(delta.imag)>1e-10)
out['DSS_nonreal_top2_product']={'count':int(bad.sum()),'negative_realpart_count':int((bad&(delta.real<0)).sum()),'rows':[{'ij':[int(i),int(j)],'alpha':float(z['alpha'][j]),'log10_temperature':float(z['log10_temperature'][i]),'multipliers':z['multipliers'][i,j].tolist(),'product':[float(delta[i,j].real),float(delta[i,j].imag)]} for i,j in zip(*np.where(bad))]}
out['DSS_negative_product_realonly']=int((ok&(abs(delta.imag)<=1e-10)&(delta.real<0)).sum())
fa=np.load(P/'fa/t-0.100/follow.npz');hp=np.load(P/'hp5/map.npz');ii=np.argmin(abs(hp['alpha2'][:,None]-fa['alpha2'][None,:]),axis=1);jj=np.argmin(abs(hp['alpha1'][:,None]-fa['alpha1'][None,:]),axis=1);inside=fa['label'][ii[:,None],jj[None,:]]>=1;per=hp['period'];ly=hp['lyapunov'];n=int(inside.sum())
out['HP5_inside_alternative_rule']={'cells':n,'positive_exponents_in_periodic':int((inside&(per>0)&(ly>1e-3)).sum()),'positive_exponents_in_family':int((inside&(per>0)&(per%5==0)&(ly>1e-3)).sum()),'positive_only_chaos':int((inside&(ly>1e-3)).sum()),'family_excluding_positive':int((inside&(per>0)&(per%5==0)&(ly<=1e-3)).sum()),'other_period_counts':{str(k):int((inside&(per==k)).sum()) for k in np.unique(per[inside&(per>0)&(per%5!=0)])}}
out['HR12_merge_shape_correction']=[]
for k in [2,5,8,14]:
 a=np.load(P/f'bands/hr12-c{k:02d}/scan-partial.npz');b=np.load(P/f'bands/hr12-c{k:02d}m/scan.npz');c=np.load(P/f'bands/hr12-c{k:02d}b/scan.npz')
 out['HR12_merge_shape_correction'].append({'band':k,'original_shape':a['period'].shape,'fields':{f:{'prefix38':bool(np.array_equal(a[f].reshape(40,640)[:38],b[f][:38],equal_nan=True)),'tail2':bool(np.array_equal(c[f],b[f][38:],equal_nan=True))} for f in ['lyapunov','period']}})
print(json.dumps(out,indent=2))
