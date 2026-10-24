"""PCP1: fixed-length prompt interpolation, map derivatives and fixed-point continuation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigs
from explore_llm_soft_loop import SoftLoop

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+1800
 def guard():
  if time.monotonic()>deadline:raise TimeoutError('PCP1 CPU wall limit')
 def record(name,value):(a.out/name).write_text(json.dumps(value,indent=2)+'\n')
 try:
  loop=SoftLoop('qwen0.5b',12,2,'open','Hello!',1,'float64','cpu');t=loop.torch;t.set_num_threads(2)
  ids=loop.prefix_ids.clone();tok=loop.sm.tok;old=tok.encode('Hello',add_special_tokens=False);new=tok.encode('Think',add_special_tokens=False)
  assert len(old)==len(new)==1
  pos=(ids[0]==old[0]).nonzero().flatten();assert len(pos)==1
  ids2=ids.clone();ids2[0,pos[0]]=new[0]
  P0=loop.E[ids].clone();P1=loop.E[ids2].clone();common=loop.prepare(np.zeros((1,896)))
  rng=np.random.default_rng(19);u=rng.normal(size=896);u/=np.linalg.norm(u);u=t.tensor(u,dtype=t.float64).reshape(1,1,-1)
  def F(x):guard();return loop.step(x,[2.0])[0]
  def J(x,v):
   guard()
   with t.no_grad():return loop.step_jvp(x,[2.0],v)[2]
  record('setup.json',dict(source=a.source,model_revision=loop.sm.engine.revision,temperature=2.,alpha=0.,dtype=str(loop.dtype),threads=2,prompt0=tok.decode(ids[0]),prompt1=tok.decode(ids2[0]),ids0=ids[0].tolist(),ids1=ids2[0].tolist(),changed_position=int(pos[0]),prefix_length=loop.L0,qualification='Exploratory soft-prompt interpolation; intermediate prompts are not discrete text.'))
  rows=[];states=[];images=[];derivatives=[];base_cache=None;previous=None
  for s in np.linspace(0,1,9):
   guard();tic=time.monotonic();loop.prefix_embeds=(1-s)*P0+s*P1;loop.prepare(np.zeros((1,896)))
   cache=[(l.keys.clone(),l.values.clone()) for l in loop.cache.layers]
   if base_cache is None:base_cache=cache
   cache_change=[float(t.sqrt(((k-bk)**2).sum()+((v-bv)**2).sum())/t.sqrt((bk**2).sum()+(bv**2).sum())) for (k,v),(bk,bv) in zip(cache,base_cache)]
   y=F(common);j=J(common,u);images.append(y.numpy());derivatives.append(j.numpy())
   fd=[]
   for h in [.1,.05,.025,.0125]:fd.append(float(((F(common+h*u)-F(common-h*u))/(2*h)-j).norm()/j.norm()))
   assert fd[-1]<1e-4 and fd[0]>fd[-1]*20,('finite difference gate',fd)
   x=common.clone() if previous is None else previous.clone();converged=False
   for iteration in range(512):
    nxt=F(x);res=float((nxt-x).norm()/x.norm());x=nxt
    if res<1e-11:converged=True;break
   row=dict(s=float(s),cache_relative_changes=cache_change,common_map_relative_change=float((y-t.as_tensor(images[0])).norm()/t.as_tensor(images[0]).norm()) if s else 0.,common_jvp_relative_change=float((j-t.as_tensor(derivatives[0])).norm()/t.as_tensor(derivatives[0]).norm()),fd_relative_errors=fd,iterations=iteration+1,converged=converged,residual=res)
   if not converged:
    rows.append(row);record('partial.json',rows);raise RuntimeError('No converged fixed point; stability analysis refused')
   previous=x.clone();states.append(x.numpy());calls=[0]
   def mv(v):
    calls[0]+=1
    return J(x,t.tensor(v,dtype=t.float64).reshape(1,1,-1)).numpy().reshape(-1)
   op=LinearOperator((896,896),matvec=mv,dtype=np.float64);runs=[]
   for seed in [13,29]:
    w,vec=eigs(op,k=3,which='LM',tol=1e-8,maxiter=100,ncv=16,v0=np.random.default_rng(seed).normal(size=896))
    order=np.argsort(-abs(w));w=w[order];vec=vec[:,order];resids=[]
    for k in range(3):
     z=vec[:,k];err=mv(z.real)+1j*mv(z.imag)-w[k]*z;resids.append(float(np.linalg.norm(err)))
    assert max(resids)<1e-6
    runs.append(dict(eigenvalues=[[float(z.real),float(z.imag)] for z in w],residuals=resids,radius=float(max(abs(w)))))
   assert abs(runs[0]['radius']-runs[1]['radius'])<1e-6
   row.update(spectra=runs,jvp_calls=calls[0],seconds=time.monotonic()-tic)
   rows.append(row);record('partial.json',rows)
   np.savez(a.out/f'point-{len(rows)-1:02d}.npz',state=x.numpy(),common_image=y.numpy(),common_jvp=j.numpy())
   print(json.dumps(row),flush=True)
  record('result.json',dict(success=True,rows=rows,seconds=time.monotonic()-start))
 except BaseException as e:record('failure.json',dict(error=repr(e),seconds=time.monotonic()-start));raise
 finally:record('manifest.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='manifest.json'})
if __name__=='__main__':main()
