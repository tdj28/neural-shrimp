"""New numerical complex-unit/flip candidate on the literal signed470 route."""
import math
import numpy as np
from neural_shrimp.unit_pair_flip import augmented,native_direction,spectrum_report,crossing_derivative
from neural_shrimp.unit_pair_scout import need,qualify,native_flat
from neural_shrimp.signed_flip_output_scout import evaluate,flat,qualify as signed_qualify


def seed_from_signed(point,M):
 z=np.asarray(point['z'],dtype=float);n=M.shape[0];need(z.shape==(2*n+1,)and np.asarray(M).shape==(n,n),'full signed phase seed')
 vals,vec=np.linalg.eig(M);i=int(np.argmax(abs(vals)));need(abs(vals[i])>1 and abs(vals[i].imag)>1e-4,'actual unstable leading complex mode')
 if vals[i].imag<0:i=int(np.argmin(abs(vals-vals[i].conjugate())))
 lam=vals[i];v=vec[:,i];need(lam.imag>0 and np.linalg.norm(M@v-lam*v)<1e-8*max(1.,np.linalg.norm(M)),'complex right-mode residual')
 a,b=v.real,v.imag;theta=.5*math.atan2(-2*float(a@b),float(a@a-b@b));candidates=[v*np.exp(1j*angle)for angle in(theta,theta+math.pi/2)];v=max(candidates,key=lambda vv:np.linalg.norm(vv.real));angle=float(np.angle(lam));scale=float(np.linalg.norm(v.real));need(0<angle<math.sqrt(3)and scale>1e-8,'regular realified complex seed')
 q=v.real/scale;r=v.imag/(angle*scale)
 if q[np.argmax(abs(q))]<0:q=-q;r=-r
 need(abs(q@q-1)<1e-12 and abs(q@r)<1e-10,'normalized orthogonal realification proposal')
 y=np.r_[z[:n],point['hidden'],point['output'],q,r,angle*angle,z[n+1:]]
 return y,dict(leading_index=i,eigenvalue=[float(lam.real),float(lam.imag)],angle=angle,q_norm=float(q@q),q_dot_r=float(q@r),scope='Finite complex-mode seed only; no unit modulus asserted before solving full55.')


def localize(model,points,seed,call,emit):
 n=model.n;k=3*n+2;y=np.asarray(seed,dtype=float).copy();left,right=[p['output']for p in points];hlo,hhi=sorted(p['hidden']for p in points)
 need(n==13 and 0<right-left<=.001 and hlo<hhi,'literal narrow output/hidden crossing bracket')
 for iteration in range(12):
  need(left-.1*(right-left)<=y[n+1]<=right+.1*(right-left)and 0<y[k]<3 and y[k+1:]@seed[k+1:]>0,'same output/angle/literal negative-mode orientation')
  v,J,data=call('augmented',y)
  if max(abs(v))<1e-11:break
  need(iteration<11,'twelve full55 return cap');step=np.linalg.solve(J,-v);emit('newton_step',dict(iteration=iteration,y=y.copy(),step=step.copy()));need(max(abs(step)/np.maximum(1.,abs(y)))<.1,'local full55 correction');y+=step
 z,t,w=y[:k],float(y[k]),y[k+1:];need(left<y[n+1]<right and hlo<y[n]<hhi,'strictly inside both literal physical-rate brackets')
 basic=qualify(z,t,v[:k],data);phase0=spectrum_report(data['monodromy'],w,math.sqrt(t));need(basic['passed']and phase0['passed'],'primitive18 and all three simple critical modes/other10stable')
 wa=data['w_phases'][1];need(np.linalg.norm(wa)>1e-8,'nonzero literal adjacent negative mode');wa=wa/np.linalg.norm(wa)
 f,g,mw,mixed,adj=call('adjacent',np.r_[data['orbit'][1],z[n:n+2],wa]);need(max(abs(f-data['orbit'][1]))<1e-9 and all(max(abs(adj['orbit'][d]-adj['orbit'][0]))>1e-5 for d in(1,2,3,6,9)),'adjacent primitive18 native return')
 phase1=spectrum_report(g[:,:n],wa,math.sqrt(t));need(phase1['passed'],'adjacent critical/simple/other10stable')
 derivative=crossing_derivative(J,w,data['monodromy'],n);need(derivative['passed'],'floating total crossing along unit-pair branch')
 signed=np.r_[y[:n],y[n],w];rv,rJ,do,rd=call('section',(signed,float(y[n+1])));need(max(abs(rv))<1e-9,'literal signed projection of full55');tangent=np.linalg.solve(rJ,-do);joined=[]
 for target in points:
  output=target['output'];u=signed+tangent*(output-y[n+1])
  for iteration in range(9):
   value,jac,partial,dd=call('section',(u,output))
   if max(abs(value))<1e-10:break
   need(iteration<8,'eight reverse corrections');delta=np.linalg.solve(jac,-value);emit('reverse_step',dict(target=target['index'],iteration=iteration,z=u.copy(),step=delta.copy()));need(max(abs(delta)/np.maximum(1.,abs(u)))<.1,'local full native reverse correction');u+=delta
  actual=flat(u,output,dd);prior=np.asarray(target['native_flat']);error=float(max(abs(actual-prior)/np.maximum(1.,abs(prior))));report=signed_qualify(u,output,value,dd);record=dict(target=target['index'],passed=bool(error<1e-7 and report['passed']),coordinates=len(actual),relative_error=error,qualification=report,native_flat=actual.tolist());emit('bracket_reverse',record);need(record['passed']and len(actual)==470,'both full470 literal bracket associations');joined.append(record)
 return dict(passed=True,y=y,native_flat=native_flat(z,data),signed_native_flat=flat(signed,float(y[n+1]),rd),native_diagnostics=basic,phase0_spectrum=phase0,phase1_spectrum=phase1,crossing_derivative=derivative,bracket_reverse=joined,scope='Second floating full55 native unitpair/flip candidate on same numerical signed route. No interval root, entire signed connection, NS coefficient or complete window claim.')
