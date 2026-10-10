"""Floating fixed-hidden-rate continuation with a free angle-squared coordinate."""
import numpy as np
from .unit_pair_scout import evaluate,qualify,native_flat,need


def section_evaluation(model,u,hidden,period=18,tick=lambda:None):
    n=model.n;z=np.asarray(u,dtype=float).copy();t=float(z[n]);z[n]=hidden
    value,jac,ht,data=evaluate(model,z,period,t,tick)
    section=jac.copy();section[:,n]=ht
    return value,section,jac[:,n].copy(),data,z,t


def native_association(flat,expected):
    flat=np.asarray(flat);expected=np.asarray(expected)
    need(flat.shape==expected.shape==(704,)and np.isfinite(flat).all(),'full native704 numerical association')
    error=float(np.max(abs(flat-expected)/np.maximum(1.,abs(expected))))
    return {'passed':error<1e-7,'scaled_max_error':error,'coordinate_count':704,'scope':'Floating same-phase/state/mode/rate comparison only, not interval branch identity.'}


def survey(model,points,*,max_points=48,max_evaluations=360,hidden_step=.25,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;need(n==13 and len(points)==20,'exact saved native prefix');calls=0
    def call(z,t):
        nonlocal calls
        need(calls<max_evaluations,'full return evaluation cap');calls+=1
        observe('evaluation_input',{'evaluation':calls,'z':np.asarray(z).tolist(),'t':t})
        try:
            v,J,ht,data=evaluate(model,z,18,t,tick)
            observe('evaluation_return',{'evaluation':calls,'z':np.asarray(z).tolist(),'t':t,'value':v.tolist(),'jacobian':J.tolist(),'parameter_derivative':ht.tolist(),'native':{k:a.tolist()for k,a in data.items()}})
            return v,J,ht,data
        except BaseException as exc:
            observe('evaluation_failure',{'evaluation':calls,'error':type(exc).__name__+': '+str(exc)});raise
    def correction(z,t,hidden=None,section=False):
        z=z.copy()
        for iteration in range(9):
            v,J,ht,data=call(z,t)
            matrix=J.copy()
            if section:matrix[:,n]=ht
            if max(abs(v))<1e-10:return z,t,v,J,ht,data
            need(iteration<8,'eight Newton updates')
            delta=np.linalg.solve(matrix,v)
            observe('newton_step',{'evaluation':calls,'section':section,'z':z.tolist(),'t':t,'step':delta.tolist()})
            scale=np.maximum(1.,abs(z));scale[n]=1. if section else scale[n]
            need(max(abs(delta)/scale)<.1,'same local Newton guard')
            if section:
                t-=float(delta[n]);delta[n]=0.;z-=delta;z[n]=hidden
            else:z-=delta
        raise AssertionError('unreachable')
    z=np.array(points[-1]['z']);t=points[-1]['t'];z,t,v,J,ht,data=correction(z,t)
    association=native_association(native_flat(z,data),points[-1]['native_flat']);need(association['passed'],'starting full native point')
    observe('reverse_anchor',association);reverse=[]
    # Reverse through the entire old t-chart prefix, including the rate turn.
    for old in reversed(points[:-1]):
        guess=z+np.linalg.solve(J,-ht)*(old['t']-t)
        nz,tt,vv,JJ,hh,dd=correction(guess,old['t']);diag=qualify(nz,tt,vv,dd,data)
        assoc=native_association(native_flat(nz,dd),old['native_flat']);need(diag['passed']and assoc['passed'],'reverse full native prefix association')
        row={'saved_index':old['index'],'z':nz.tolist(),'t':tt,'diagnostics':diag,'association':assoc,'native_flat':native_flat(nz,dd).tolist()};observe('reverse_point',row);reverse.append(row)
        z,t,v,J,ht,data=nz,tt,vv,JJ,hh,dd
    # Restart the exact saved last point only after its entire reverse chain passed.
    z=np.array(points[-1]['z']);t=points[-1]['t'];z,t,v,J,ht,data=correction(z,t)
    need(native_association(native_flat(z,data),points[-1]['native_flat'])['passed'],'same reparameterization anchor')
    accepted=[];previous=None
    for index in range(max_points):
        diag=qualify(z,t,v,data,previous);need(diag['passed'],'qualified numerical native section point')
        ellipse=float(z[n+2:2*n+2]@z[n+2:2*n+2]-t*(z[2*n+2:]@z[2*n+2:]))
        row={'index':index,'z':z.tolist(),'t':t,'diagnostics':diag,'native_flat':native_flat(z,data).tolist(),'ellipticity_gauge_factor':ellipse};observe('point',row);accepted.append(row)
        common={'points':accepted,'reverse_prefix':reverse,'evaluations':calls}
        if t<1e-5:return dict(common,status='near_double_plus_proposal')
        if abs(ellipse)<.05:return dict(common,status='near_gauge_degeneracy_proposal')
        if diag['complement_unit_gap']<.03:return dict(common,status='competing_unit_mode_proposal')
        if index==max_points-1:return dict(common,status='point_cap')
        matrix=J.copy();matrix[:,n]=ht;tangent=np.linalg.solve(matrix,-J[:,n]);step=hidden_step
        observe('section_tangent',{'evaluation':calls,'z':z.tolist(),'t':t,'tangent':tangent.tolist(),'coordinate_convention':'index13 is t, physical hidden rate is independent'})
        success=False
        for retry in range(5):
            hidden=float(z[n]+step);guess=z+tangent*step;tt=t+float(tangent[n])*step;guess[n]=hidden
            record={'index':index,'retry':retry,'hidden':hidden,'predictor':guess.tolist(),'t_predictor':tt,'passed':False}
            try:
                zz,tt,vv,JJ,hh,dd=correction(guess,tt,hidden,True);di=qualify(zz,tt,vv,dd,data)
                need(di['passed']and max(abs(zz-z)/np.maximum(1.,abs(z)))<.1,'native local section association')
                # Reverse in the same physical section to the exact preceding point.
                backmatrix=JJ.copy();backmatrix[:,n]=hh;back=np.linalg.solve(backmatrix,-JJ[:,n]);rz=zz-back*step;rt=tt-float(back[n])*step;rz[n]=z[n]
                rz,rt,rv,rJ,rh,rd=correction(rz,rt,float(z[n]),True)
                reverse_assoc=native_association(native_flat(rz,rd),native_flat(z,data));need(reverse_assoc['passed']and abs(rt-t)<1e-7 and qualify(rz,rt,rv,rd)['passed'],'full native reverse section join')
                record.update(passed=True,diagnostics=di,reverse=reverse_assoc,reverse_t=rt,reverse_z=rz.tolist())
                previous=data;z,t,v,J,ht,data=zz,tt,vv,JJ,hh,dd;success=True
            except (ValueError,np.linalg.LinAlgError,FloatingPointError)as exc:record['error']=type(exc).__name__+': '+str(exc)
            observe('attempt',record)
            if success:break
            step/=2
        if not success:return dict(common,status='corrector_limit',evaluations=calls)
    raise AssertionError('unreachable')
