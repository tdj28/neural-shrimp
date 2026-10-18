"""Own-float64 map adapter to disclosed minimally augmented continuation code.
Prepared source only: an exact input/profile card is required before target use.
"""
import argparse,importlib.util,json,sys,time,hashlib
from pathlib import Path
import numpy as np

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();c=json.loads(Path(a.card).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 for pin in c['inputs']:
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 q=load('curve_own_map',c['map_source']);bc=load('curve_algorithm',c['continuation_source']);model=q.ManualQwenSoftMap.from_pinned_files();period=c['period'];assert type(period)is int and period>=1
 start=time.monotonic();counts={'dense':0,'jvp':0,'grad':0,'param':0,'residual':0};events=(out/'events.jsonl').open('x');records=[]
 def event(stage,**v):events.write(json.dumps({'elapsed':time.monotonic()-start,'stage':stage,'counts':counts.copy(),**v})+'\n');events.flush()
 # Physical parameters are theta=(log10T, 10 alpha). No detached prefix in derivatives.
 def phi(x,t):return model.period_map(x,t[1]/10,t[0],period)
 class OwnFamily(bc.Family):
  def __init__(self):super().__init__(phi,(896,),torch.float64);self.latest=None
  def residual(self,x,t):
   counts['residual']+=1;assert counts['residual']<=c['max_residuals'];return super().residual(x,t)
  def monodromy(self,x,t):
   counts['dense']+=1;assert counts['dense']<=c['max_dense']
   M=model.jacobian(torch.from_numpy(np.asarray(x)),t[1]/10,t[0],period=period,chunk_size=16).detach().numpy();assert np.isfinite(M).all();self.latest=(np.array(x).copy(),np.array(t).copy(),M.copy());event('dense');return M
  def jvp_x(self,x,t,v):
   counts['jvp']+=1;assert counts['jvp']<=c['max_jvps'];return super().jvp_x(x,t,v)
  def grad_s(self,x,t,v,w):
   counts['grad']+=1;assert counts['grad']<=c['max_grad'];return super().grad_s(x,t,v,w)
  def param_jacobian(self,x,t,which=(0,1)):
   counts['param']+=1;assert counts['param']<=c['max_param'];return super().param_jacobian(x,t,which)
 family=OwnFamily();x=np.load(c['state'],allow_pickle=False);assert x.shape==(896,) and np.isfinite(x).all();theta=np.array(c['theta'],float);kind=c['kind'];assert kind in ['fold','flip'];target=1 if kind=='fold' else -1
 def precision(x,t,M):
  v=np.arange(1,897,dtype=float);v/=np.linalg.norm(v);ad=family.jvp_x(x,t,v);den=np.linalg.norm(ad);assert den>0
  rows=[]
  with torch.no_grad():
   for h in [.01,.003,.001]:
    fp=phi(torch.from_numpy(x+h*v),torch.from_numpy(t)).numpy();fm=phi(torch.from_numpy(x-h*v),torch.from_numpy(t)).numpy();rows.append({'h':h,'relative_error':float(np.linalg.norm((fp-fm)/(2*h)-ad)/den)})
  error=float(np.linalg.norm(M@v-ad)/den);return {'dense_jvp_error':error,'fd':rows,'passed':error<1e-8 and rows[-1]['relative_error']<rows[0]['relative_error']/10 and rows[-1]['relative_error']<1e-3}
 def on_point(row):
  xx,tt,M=family.latest;assert np.array_equal(tt,np.array(row['theta']))
  vals,vecs=np.linalg.eig(M);res=np.linalg.norm(M@vecs-vecs*vals,axis=0)/np.linalg.norm(vecs,axis=0);idx=int(np.argmin(abs(vals-target)));mu=vals[idx];rest=np.delete(vals,idx);returns=[];y=torch.from_numpy(xx.copy());cache=model.prefix_cache(tt[1]/10)
  with torch.no_grad():
   for k in range(1,period+1):y=model(y,tt[1]/10,tt[0],cache);returns.append(float(np.linalg.norm(y.numpy()-xx)/np.linalg.norm(xx)))
  checks={'closure':returns[-1]<1e-10,'primitive':all(returns[d-1]>1e-6 for d in range(1,period) if period%d==0),'real_target':abs(mu.imag)<1e-8 and abs(mu.real-target)<1e-6,'isolated_target':bool(np.min(abs(rest-target))>1e-5),'eigen_residual':bool(np.max(res)<1e-8),'g':row['g']<1e-9 and row['g_dense']<1e-9,'krylov':row['gmres_unconverged']==0}
  rr={'producer_algorithm_row':row,'returns':returns,'critical_modulus':float(abs(mu)),'critical_argument':float(np.angle(mu)),'next_modulus':float(np.max(abs(rest))),'multipliers':[[float(v.real),float(v.imag)] for v in vals],'eigen_residuals':res.tolist(),'checks':checks}
  if not records:rr['precision']=precision(xx,tt,M);checks['composed_precision']=rr['precision']['passed']
  rr['qualified']=all(checks.values());n=len(records);np.savez(out/f'point-{n:03d}.npz',state=xx,theta=tt,monodromy=M,eigenvalues=vals,eigenvectors=vecs);records.append(rr);(out/'points.json').write_text(json.dumps(records,indent=2)+'\n');event('point',index=n,qualified=rr['qualified'])
  if not rr['qualified']:raise ValueError('point qualification failed; retained and continuation stopped')
 try:
  if c['mode']=='profile':
   M=family.monodromy(x,theta);pr=precision(x,theta,M);b,cc,_=bc.initial_borders(M,kind);v,g,w=bc.bordered_solve(M-target*np.eye(896),b,cc);gx,gt=family.grad_s(x,theta,v,w);dp=family.param_jacobian(x,theta);np.savez(out/'profile.npz',state=x,theta=theta,monodromy=M,grad_x=gx,grad_theta=gt,parameter_jacobian=dp);report={'completed':True,'mode':'profile','precision':pr,'scope':'cost and precision only; not a crossing or continuation result'}
  else:
   assert c['mode']=='continue';rr,reason,states=bc.trace_curve(family,x,theta,kind,fix=0,direction=c['direction'],steps=c['points'],ds=c['ds'],ds_min=c['ds_min'],ds_max=c['ds_max'],tol=1e-10,gtol=1e-9,max_newton=6,bounds=c['bounds'],on_point=on_point,k=896,chord=True,gmres_tol=1e-9,gmres_maxiter=2);report={'completed':True,'stop_reason':reason,'qualified_points':len(records),'records':records,'scope':'Numerical full-state continuation, not an interval geometry certificate.'}
 except Exception as e:
  report={'completed':False,'failure':repr(e),'retained_points':len(records)};(out/'failure.json').write_text(json.dumps(report,indent=2)+'\n');raise
 for pin in c['inputs']:
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
 report.update(elapsed=time.monotonic()-start,counts=counts);(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
