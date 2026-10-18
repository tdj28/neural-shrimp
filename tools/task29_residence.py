"""Saved-saddle residence experiment; all scales/radii/sides retained."""
import argparse,json,importlib.util,sys,time,resource
from pathlib import Path
import numpy as np

def residence(distances,radius):
 inside=distances<radius;runs=[];start=None
 for i,value in enumerate(inside):
  if value and start is None:start=i
  if start is not None and (not value or i==len(inside)-1):
   end=i if not value else i+1;runs.append(dict(entry=start,exit=None if value else i,duration=end-start,censored=bool(value)));start=None
 return runs

def fit_scales(d,t):
 if len(d)<8 or np.log(max(d)/min(d))<np.log(100):return {'qualified':False,'reason':'fewer than8 uncensored scales or less than2 decades'}
 slope,intercept=np.polyfit(np.log(d),t,1);res=np.array(t)-(slope*np.log(d)+intercept)
 return {'qualified':True,'n':len(d),'slope':float(slope),'intercept':float(intercept),'rmse':float(np.sqrt(np.mean(res**2))),'range':[min(d),max(d)],'residuals':res.tolist()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();c=json.loads(Path(args.card).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);spec=importlib.util.spec_from_file_location('residence_map',c['map_source']);q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
 for e in c['inputs']:assert q.sha256(e['path'])==e['sha256']
 saddle=json.loads(Path(c['saddle_report']).read_text());assert saddle['saddle_qualified'];orbit=np.load(c['orbit'],allow_pickle=False)[:-1];p=saddle['period'];assert len(orbit)==p;norm=float(np.mean(np.linalg.norm(orbit,axis=1)));assert norm>0
 ref=np.load(c['references'],allow_pickle=False)['tail'];left,right=[ref[-1,j] for j in c['reference_columns']];lo,hi=c['bracket'];star=(lo+hi)/2;width=hi-lo;assert 0<lo<hi<1
 lower=max(64*width,1e-9);upper=min(.02,star/2,(1-star)/2);assert upper/lower>=100,'bracket too broad for two-decade residence test'
 distances=np.geomspace(lower,upper,16);us=np.concatenate([star-distances,star+distances]);starts=(1-us[:,None])*left+us[:,None]*right
 # The entire trajectory is retained for a separate saved-distance audit.
 model=q.ManualQwenSoftMap.from_pinned_files();cache=model.prefix_cache(c['alpha']);x=torch.as_tensor(starts,dtype=torch.float64);mem=np.lib.format.open_memmap(out/'states.npy',mode='w+',dtype=np.float64,shape=(c['steps']+1,32,896));mem[0]=starts;tic=time.monotonic();events=(out/'events.jsonl').open('x')
 with torch.no_grad():
  for n in range(1,c['steps']+1):
   x=model(x,c['alpha'],c['log10T'],cache);assert torch.isfinite(x).all();mem[n]=x.numpy()
   if n%100==0:events.write(json.dumps(dict(step=n,elapsed=time.monotonic()-tic))+'\n');events.flush();mem.flush()
 mem.flush();all_distances=np.empty((c['steps']+1,32))
 for i in range(len(mem)):all_distances[i]=np.min(np.linalg.norm(mem[i,:,None,:]-orbit[None,:,:],axis=-1),axis=1)/norm
 np.save(out/'phase-distances.npy',all_distances);rows=[];fits=[];coalesced=set()
 for side in range(2):
  for j in range(15):
   k=side*16+j
   if np.array_equal(mem[1:65,k],mem[1:65,k+1]):coalesced.update([k,k+1])
 for rho in [3e-4,1e-3,3e-3]:
  for side in range(2):
   ds=[];ts=[];indices=[]
   for j,d in enumerate(distances):
    k=side*16+j;runs=residence(all_distances[:,k],rho);initial_separation=np.linalg.norm(starts[k]-((1-star)*left+star*right))/norm;roundoff=initial_separation<=1e-11 or k in coalesced
    row={'radius':rho,'side':-1 if side==0 else 1,'index':k,'distance_u':float(d),'s':float(us[k]),'bracket_center':star,'bracket_width':width,'normalized_initial_separation':float(initial_separation),'roundoff_excluded':bool(roundoff),'exact_early_trajectory_coalescence':k in coalesced,'runs':runs};rows.append(row)
    if runs and not runs[0]['censored'] and not roundoff:ds.append(float(d));ts.append(runs[0]['duration']);indices.append(k)
   fit=fit_scales(ds,ts);fit.update(radius=rho,side=-1 if side==0 else 1,indices=indices,predicted=saddle['predicted_escape_slope'])
   if fit['qualified']:fit['signed_relative_slope_difference']=(fit['slope']-fit['predicted'])/abs(fit['predicted'])
   fit['inner_range_sensitivity']=fit_scales(ds[:-4],ts[:-4]) if len(ds)>4 else {'qualified':False}
   fit['outer_range_sensitivity']=fit_scales(ds[4:],ts[4:]) if len(ds)>4 else {'qualified':False};fits.append(fit)
 for e in c['inputs']:assert q.sha256(e['path'])==e['sha256']
 report={'completed':True,'rows':rows,'fits':fits,'elapsed':time.monotonic()-tic,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Deterministic geometric scales, not IID samples. Primary residence is first contiguous visit at radius1e-3; two radii and range sensitivities retained. Failed entry/censoring/roundoff remain explicit. No automatic statistical CI or causal global claim. Bracket-center error <=distance/128; finite bracket may contain multiple switches.'}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
