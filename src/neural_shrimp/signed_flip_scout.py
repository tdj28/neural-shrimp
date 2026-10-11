"""Numerical full native signed-locus proposals, fixed physical hidden rate."""
import numpy as np
from neural_shrimp.unit_pair_scout import Jet,need

def evaluate(model,z,hidden,period=18,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;z=np.asarray(z,dtype=float);need(z.shape==(2*n+1,)and np.isfinite(z).all()and np.isfinite(hidden),'full x/output/mode dimensions')
    x=z[:n];output=z[n];q=z[n+1:];base=n+2
    state=[Jet.variable(x[i],base,i,q[i])for i in range(n)];h=Jet.variable(hidden,base,n);o=Jet.variable(output,base,n+1)
    orbit=[x.copy()];modes=[q.copy()];local=[]
    for k in range(period):
        observe('native_input',dict(phase=k,values=[[v.f,v.g.tolist(),v.d,v.m.tolist()]for v in state+[h,o]]));tick();state=model.cusp_step(state,h,o);ret=[[v.f,v.g.tolist(),v.d,v.m.tolist()]for v in state];observe('native_return',dict(phase=k,values=ret));orbit.append(np.array([v.f for v in state]));modes.append(np.array([v.d for v in state]));local.append(ret)
    M=np.array([v.g[:n]for v in state]);value=np.r_[[v.f for v in state]-x,[v.d for v in state]+q,(q@q-1)/2]
    J=np.zeros((2*n+1,2*n+1));J[:n,:n]=M-np.eye(n);J[:n,n]=[v.g[n+1]for v in state];J[n:2*n,:n]=[v.m[:n]for v in state];J[n:2*n,n]=[v.m[n+1]for v in state];J[n:2*n,n+1:]=M+np.eye(n);J[-1,n+1:]=q
    dh=np.r_[[v.g[n]for v in state],[v.m[n]for v in state],0.]
    need(np.isfinite(value).all()and np.isfinite(J).all()and np.isfinite(dh).all(),'finite full native return')
    return value,J,dh,dict(orbit=np.array(orbit),modes=np.array(modes),monodromy=M,native_returns=local)

def flat(z,h,data):return np.r_[np.concatenate([np.r_[x,q]for x,q in zip(data['orbit'][:-1],data['modes'][:-1])]),h,z[data['monodromy'].shape[0]]]

def qualify(z,h,value,data,prior=None):
    n=data['monodromy'].shape[0];p=len(data['orbit'])-1;M=data['monodromy'];vals,vecs=np.linalg.eig(M);i=int(np.argmin(abs(vals+1)));others=[v for j,v in enumerate(vals)if j!=i]
    errors=[float(np.linalg.norm(M@vecs[:,j]-vals[j]*vecs[:,j])/max(1.,np.linalg.norm(M)*np.linalg.norm(vecs[:,j])))for j in range(n)]
    div={str(k):float(np.max(abs(data['orbit'][k]-data['orbit'][0])))for k in range(1,p)if p%k==0};jump=0. if prior is None else float(np.max(abs(data['orbit']-prior['orbit'])/np.maximum(1.,abs(prior['orbit']))))
    gap=min((abs(v+1)for v in others),default=1.)
    gates=dict(residual=bool(max(abs(value))<1e-9),positive_rates=bool(h>0 and z[n]>0),primitive=all(v>1e-5 for v in div.values()),minus_one=bool(abs(vals[i]+1)<1e-7),simple=bool(gap>1e-5),spectrum=bool(max(errors)<1e-9),native_step=bool(jump<.1))
    return dict(passed=all(gates.values()),gates=gates,divisors=div,eigenvalues=[[float(v.real),float(v.imag)]for v in vals],eigen_residuals=errors,critical_index=i,critical_gap=float(gap),complement_max_modulus=float(max(map(abs,others),default=0.)),plus_one_distance=float(min((abs(v-1)for v in others),default=1.)),native_step=jump,scope='Numerical proposals only; exact same-branch and finite stability require independent certificates.')

def survey(model,seed,hidden,*,step=-.25,max_points=48,max_evaluations=320,tick=lambda:None,observe=lambda k,v:None):
    """Retained local corrections and all-coordinate reverse associations."""
    calls=0;points=[];n=model.n
    def call(z,h):
        nonlocal calls

        if calls>=max_evaluations:raise RuntimeError('fixed full return cap')
        calls+=1;observe('evaluation_input',dict(index=calls,z=np.asarray(z).tolist(),hidden=float(h)))
        try:
            v,J,dh,d=evaluate(model,z,h,18,tick,observe);observe('evaluation_return',dict(index=calls,value=v.tolist(),jacobian=J.tolist(),parameter_derivative=dh.tolist(),native={k:(x.tolist()if isinstance(x,np.ndarray)else x)for k,x in d.items()}));return v,J,dh,d
        except BaseException as e:observe('evaluation_failure',dict(index=calls,error=type(e).__name__+': '+str(e)));raise
    def correct(z,h):
        z=np.array(z,dtype=float)
        for it in range(9):
            v,J,dh,d=call(z,h)
            if max(abs(v))<1e-10:return z,v,J,dh,d
            need(it<8,'local corrector limit');delta=np.linalg.solve(J,-v);observe('newton_step',dict(evaluation=calls,iteration=it,step=delta.tolist()));need(max(abs(delta)/np.maximum(1.,abs(z)))<.1,'local fullstate correction');z+=delta
        raise AssertionError('unreachable')
    z,v,J,dh,d=correct(seed,hidden);previous=None
    for index in range(max_points):
        info=qualify(z,hidden,v,d,previous);need(info['passed'],'all native qualification gates');row=dict(index=index,z=z.tolist(),hidden=hidden,diagnostics=info,native_flat=flat(z,hidden,d).tolist());points.append(row);observe('accepted_point',row)
        base=dict(points=points,evaluations=calls)
        if index and (points[-2]['diagnostics']['complement_max_modulus']-1)*(info['complement_max_modulus']-1)<=0:return dict(base,status='noncritical_unit_crossing_proposal')
        if info['critical_gap']<.01:return dict(base,status='near_competing_minus_one_proposal')
        if info['plus_one_distance']<.01:return dict(base,status='near_plus_one_proposal')
        if index==max_points-1:return dict(base,status='point_cap')
        tangent=np.linalg.solve(J,-dh);observe('tangent_proposal',dict(index=index,tangent=tangent.tolist()));accepted=False;dhstep=step
        for retry in range(5):
            hh=hidden+dhstep;guess=z+tangent*dhstep;record=dict(index=index,retry=retry,hidden=hh,predictor=guess.tolist(),passed=False)
            try:
                zz,vv,JJ,ddh,dd=correct(guess,hh);di=qualify(zz,hh,vv,dd,d);need(di['passed']and max(abs(zz-z)/np.maximum(1.,abs(z)))<.1,'full-state neighboring branch guard')
                bt=np.linalg.solve(JJ,-ddh);rz,rv,rJ,rh,rd=correct(zz-bt*dhstep,hidden);a=flat(z,hidden,d);b=flat(rz,hidden,rd);error=float(np.max(abs(a-b)/np.maximum(1.,abs(a))));need(error<1e-7 and qualify(rz,hidden,rv,rd)['passed'],'all native470 reverse association');record.update(passed=True,reverse_error=error,reverse_coordinates=len(a));previous=d;z,v,J,dh,d=zz,vv,JJ,ddh,dd;hidden=hh;accepted=True
            except (ValueError,np.linalg.LinAlgError,FloatingPointError)as e:record['error']=type(e).__name__+': '+str(e)
            observe('continuation_attempt',record)
            if accepted:break
            dhstep/=2
        if not accepted:return dict(points=points,evaluations=calls,status='corrector_limit')
    raise AssertionError('unreachable')
