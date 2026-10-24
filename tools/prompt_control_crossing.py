"""PCP1 exploratory refinement of a real-minus-one stability bracket."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigs
from explore_llm_soft_loop import SoftLoop
from prompt_control_probe import newton_fixed_point

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);deadline=time.monotonic()+1200
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def record(n,v):(a.out/n).write_text(json.dumps(v,indent=2)+'\n')
 try:
  for n,h in json.loads((a.input/'manifest.json').read_text()).items():assert sha(a.input/n)==h
  rows=json.loads((a.input/'partial.json').read_text());setup=json.loads((a.input/'setup.json').read_text())
  def mu(row):
   z=complex(*row['spectra'][0]['eigenvalues'][0]);assert abs(z.imag)<1e-8 and z.real<0
   return z.real
  left,right=rows[5],rows[6];assert left['s']==.625 and right['s']==.75 and mu(left)>-1>mu(right)
  l,r=.625,.75;xl=np.load(a.input/'point-05.npz')['state'];xr=np.load(a.input/'point-06.npz')['state']
  loop=SoftLoop('qwen0.5b',12,2,'open','Hello!',1,'float64','cpu');t=loop.torch;t.set_num_threads(2)
  P0=loop.E[t.tensor([setup['ids0']])];P1=loop.E[t.tensor([setup['ids1']])]
  record('setup.json',dict(source=a.source,parent=str(a.input),parent_manifest_sha256=sha(a.input/'manifest.json'),initial_bracket=[l,r],steps=4))
  def guard():
   if time.monotonic()>deadline:raise TimeoutError('Crossing CPU time limit')
  def F(x):guard();return loop.step(x,[2.])[0]
  def J(x,v):
   guard()
   with t.no_grad():return loop.step_jvp(x,[2.],v)[2]
  history=[]
  for i in range(4):
   s=(l+r)/2;loop.prefix_embeds=(1-s)*P0+s*P1;loop.prepare(np.zeros((1,896)))
   x,its,res=newton_fixed_point(t.tensor((xl+xr)/2,dtype=t.float64),F,J)
   def mv(v):return J(x,t.tensor(v,dtype=t.float64).reshape(1,1,-1)).numpy().reshape(-1)
   op=LinearOperator((896,896),matvec=mv,dtype=np.float64)
   vals,vec=eigs(op,k=3,ncv=16,tol=1e-9,maxiter=100,v0=np.random.default_rng(13).normal(size=896));order=np.argsort(-abs(vals));vals=vals[order];vec=vec[:,order]
   residuals=[]
   for k in range(3):
    z=vec[:,k];residuals.append(float(np.linalg.norm(mv(z.real)+1j*mv(z.imag)-vals[k]*z)))
   assert max(residuals)<1e-6
   assert abs(vals[0].imag)<1e-8 and vals[0].real<0 and abs(vals[1])<1
   row=dict(s=s,multiplier=float(vals[0].real),modulus=float(abs(vals[0])),argument=float(np.angle(vals[0])),next_largest_modulus=float(abs(vals[1])),eigen_residuals=residuals,fixed_point_residual=res,newton_iterations=its)
   history.append(row);np.savez(a.out/f'point-{i:02d}.npz',state=x.numpy());record('partial.json',history);print(json.dumps(row),flush=True)
   if vals[0].real>-1:l=s;xl=x.numpy()
   else:r=s;xr=x.numpy()
  record('result.json',dict(success=True,bracket=[l,r],width=r-l,history=history,qualification='Numerical real-minus-one stability bracket; not interval-certified and no flip normal-form coefficient or stable doubled orbit computed.'))
 except BaseException as e:record('failure.json',dict(error=repr(e)));raise
 finally:record('manifest.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='manifest.json'})
if __name__=='__main__':main()
