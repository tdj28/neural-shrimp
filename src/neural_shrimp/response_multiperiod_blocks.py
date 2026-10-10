"""Fixed two- or four-period block Jacobi for saved finite-window response operators."""
import hashlib,time
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from scipy.sparse.linalg import LinearOperator,cg
from neural_shrimp.response_spectrum import checked,difference_operator

def require(ok,message):
 if not ok:raise ValueError(message)

def block_jacobi(maps,steps,shift,eta,block_periods,emit=lambda k,v:None,tick=lambda:None):
 a=checked(maps)*np.exp(-shift);p,n,_=a.shape;require(steps%p==0 and eta>0 and np.isfinite(eta),'whole period positive regularizer');require(type(block_periods)is int and block_periods in (2,4),'fixed block periods2or4');chunk=p*block_periods;require(steps%chunk==0,'whole chunks');cache={};blocks=[];stats=dict(cholesky=0,applications=0,triangular_solves=0)
 for start in range(0,steps+1,chunk):
  tick();end=min(start+chunk,steps+1);size=(end-start)*n;h=np.zeros((size,size))
  for t in range(start,end):
   i=(t-start)*n;diag=eta**2*np.eye(n)
   if t>0:diag+=np.eye(n)
   if t<steps:diag+=a[t%p].T@a[t%p]
   h[i:i+n,i:i+n]=diag
   if t+1<end:
    h[i:i+n,i+n:i+2*n]=-a[t%p].T;h[i+n:i+2*n,i:i+n]=-a[t%p]
  key=hashlib.sha256(h.tobytes()).hexdigest();emit('block_input',dict(start=start,end=end,matrix=h,key=key))
  if key not in cache:
   stats['cholesky']+=1;factor=cholesky(h,lower=True);emit('factor',dict(key=key,factor=factor,matrix=h));require(np.isfinite(factor).all(),'finite Cholesky');error=np.linalg.norm(factor@factor.T-h)/max(np.linalg.norm(h),1e-300);emit('factor_check',dict(key=key,relative_residual=float(error)));require(error<=1e-10,'factor residual');cache[key]=factor
  blocks.append((start*n,end*n,key))
 def apply(v):
  tick();stats['applications']+=1;require(stats['applications']<=301,'preconditioner application cap');out=np.empty_like(v)
  for start,end,key in blocks:
   stats['triangular_solves']+=2;z=solve_triangular(cache[key],v[start:end],lower=True);out[start:end]=solve_triangular(cache[key].T,z,lower=False)
  require(np.isfinite(out).all(),'finite preconditioner return');return out
 return LinearOperator(((steps+1)*n,)*2,matvec=apply,dtype=float),stats,blocks

def probe(maps,block_periods,emit=lambda k,v:None,tick=lambda:None):
 a=checked(maps);p,n,_=a.shape;steps=16*p;shift=-.05;eta=.01;counts={};start=time.monotonic();actions=[lambda v,m=m:m@v for m in a];adjoints=[lambda v,m=m:m.T@v for m in a];k,counts=difference_operator(actions,adjoints,n,steps,shift,'free',tick);normal_calls=[0]
 def normal(v):
  tick();normal_calls[0]+=1;require(normal_calls[0]<=302,'normal action cap');return k.T@(k@v)+eta**2*v
 operator=LinearOperator(((steps+1)*n,)*2,matvec=normal,dtype=float);impulse=np.zeros(steps*n);time0=8*p;impulse[time0*n]=1.;rhs=k.T@impulse;emit('rhs',dict(impulse=impulse,rhs=rhs));pre=None;stats={};blocks=[];setup=time.monotonic()
 pre,stats,blocks=block_jacobi(a,steps,shift,eta,block_periods,emit,tick)
 setup_seconds=time.monotonic()-setup;iterations=[0]
 def callback(v):
  iterations[0]+=1;emit('iterate',dict(iteration=iterations[0],state=v.copy()));tick()
 v,info=cg(operator,rhs,x0=np.zeros_like(rhs),rtol=1e-10,atol=0.,maxiter=300,M=pre,callback=callback);emit('solution',dict(state=v,info=int(info)));require(np.isfinite(v).all(),'finite solution');res=k@v-impulse;true_normal=k.T@res+eta**2*v;emit('actual_residual',dict(data_residual=res,normal_residual=true_normal));relative=float(np.linalg.norm(true_normal)/np.linalg.norm(rhs));readout=v[(time0+1)*n:(time0+2)*n].copy();tick()
 return dict(method=str(block_periods)+'_period_block_pcg',block_periods=block_periods,info=int(info),iterations=iterations[0],counts=counts,normal_calls=normal_calls[0],preconditioner=stats,blocks=blocks,setup_seconds=setup_seconds,elapsed_seconds=time.monotonic()-start,true_normal_residual_norm=float(np.linalg.norm(true_normal)),true_normal_relative_residual=relative,solve_qualified=bool(info==0 and relative<=1e-8),readout_column=readout.tolist(),settings=dict(periods=16,shift=shift,eta=eta,boundary='free',maxiter=300,rtol=1e-10,impulse='e0 at central phase0'),scope='saved small-matrix callback arithmetic; one coordinate column, not trace or spectral count; exact small-block factor oracle, no measured large-model HVP speedup')
