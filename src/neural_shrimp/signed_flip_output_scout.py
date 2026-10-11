"""Fixed-output numerical native minus-one section; same full470 coordinates."""
import numpy as np
from . import signed_flip_scout as S
from .unit_pair_scout import need


def evaluate(model,z,output,period=18,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;z=np.asarray(z,dtype=float)
    need(z.shape==(2*n+1,)and np.isfinite(z).all()and np.isfinite(output),'full output section dimensions')
    native=np.r_[z[:n],output,z[n+1:]]
    value,J,dh,data=S.evaluate(model,native,z[n],period,tick,observe)
    partial=J[:,n].copy();section=J.copy();section[:,n]=dh
    return value,section,partial,data


def flat(z,output,data):
    n=data['monodromy'].shape[0]
    return S.flat(np.r_[z[:n],output,z[n+1:]],z[n],data)


def qualify(z,output,value,data,prior=None):
    n=data['monodromy'].shape[0]
    return S.qualify(np.r_[z[:n],output,z[n+1:]],z[n],value,data,prior)


def survey(model,seed,output,*,step=.000025,max_points=32,max_evaluations=240,near_plus_threshold=.01,tick=lambda:None,observe=lambda k,v:None):
    """Retained local corrections and all-coordinate reverse associations."""
    need(np.isfinite(near_plus_threshold)and 0<near_plus_threshold<=.01,'bounded near-plus event threshold')
    calls=0;points=[];n=model.n
    def call(z,h):
        nonlocal calls

        if calls>=max_evaluations:raise RuntimeError('fixed full return cap')
        calls+=1;observe('evaluation_input',dict(index=calls,z=np.asarray(z).tolist(),output=float(h)))
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
    z,v,J,dh,d=correct(seed,output);previous=None
    for index in range(max_points):
        info=qualify(z,output,v,d,previous);need(info['passed'],'all native qualification gates');row=dict(index=index,z=z.tolist(),output=output,hidden=float(z[n]),diagnostics=info,native_flat=flat(z,output,d).tolist());points.append(row);observe('accepted_point',row)
        base=dict(points=points,evaluations=calls)
        if index and (points[-2]['diagnostics']['complement_max_modulus']-1)*(info['complement_max_modulus']-1)<=0:return dict(base,status='noncritical_unit_crossing_proposal')
        if info['critical_gap']<.01:return dict(base,status='near_competing_minus_one_proposal')
        if info['plus_one_distance']<near_plus_threshold:return dict(base,status='near_plus_one_proposal')
        if index==max_points-1:return dict(base,status='point_cap')
        tangent=np.linalg.solve(J,-dh);observe('tangent_proposal',dict(index=index,tangent=tangent.tolist()));accepted=False;dhstep=step
        for retry in range(5):
            hh=output+dhstep;guess=z+tangent*dhstep;record=dict(index=index,retry=retry,output=hh,predictor=guess.tolist(),passed=False)
            try:
                zz,vv,JJ,ddh,dd=correct(guess,hh);di=qualify(zz,hh,vv,dd,d);need(di['passed']and max(abs(zz-z)/np.maximum(1.,abs(z)))<.1,'full-state neighboring branch guard')
                bt=np.linalg.solve(JJ,-ddh);rz,rv,rJ,rh,rd=correct(zz-bt*dhstep,output);a=flat(z,output,d);b=flat(rz,output,rd);error=float(np.max(abs(a-b)/np.maximum(1.,abs(a))));need(error<1e-7 and qualify(rz,output,rv,rd)['passed'],'all native470 reverse association');record.update(passed=True,reverse_error=error,reverse_coordinates=len(a));previous=d;z,v,J,dh,d=zz,vv,JJ,ddh,dd;output=hh;accepted=True
            except (ValueError,np.linalg.LinAlgError,FloatingPointError)as e:record['error']=type(e).__name__+': '+str(e)
            observe('continuation_attempt',record)
            if accepted:break
            dhstep/=2
        if not accepted:return dict(points=points,evaluations=calls,status='corrector_limit')
    raise AssertionError('unreachable')
