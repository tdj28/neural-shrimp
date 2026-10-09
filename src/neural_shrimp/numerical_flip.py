"""Bounded numerical flip exploration; all claims are floating-point only."""
import numpy as np
from scipy.optimize import root


def spectrum(matrix):
    values,vectors=np.linalg.eig(matrix)
    scale=max(1.,float(np.linalg.norm(matrix,np.inf)))
    residuals=[float(np.linalg.norm(matrix@vectors[:,i]-values[i]*vectors[:,i],np.inf)/(scale*np.linalg.norm(vectors[:,i],np.inf))) for i in range(len(values))]
    return values,vectors,residuals


def qualify(data,closure_tol=1e-8,divisor_margin=1e-6):
    x=np.asarray(data['orbit'][0]);p=data['period'];orbit=np.asarray(data['orbit']);m=np.asarray(data['monodromy'])
    scale=max(1.,float(np.max(abs(x))))
    closure=float(np.max(abs(orbit[-1]-x))/scale)
    divisions=[{'divisor':d,'separation':float(np.max(abs(orbit[d]-x))/scale)} for d in range(1,p) if p%d==0]
    values,vectors,residuals=spectrum(m)
    radius=float(max(abs(values)))
    gates={'closure':closure<=closure_tol,'primitive':all(d['separation']>=divisor_margin for d in divisions),'spectral_residuals':max(residuals)<1e-9}
    return {'qualified':all(gates.values()),'gates':gates,'closure':closure,'divisors':divisions,
        'spectral_radius':radius,'stability':'attracting' if radius<1-1e-7 else 'unstable' if radius>1+1e-7 else 'neutral_or_unresolved',
        'multipliers':[[float(v.real),float(v.imag)] for v in values],'eigen_residuals':residuals}


def correct(evaluate,seed,s,period,max_iterations=24):
    x=np.array(seed,dtype=float);history=[];last=None
    for iteration in range(max_iterations):
        data=evaluate(x,s,period);last=data
        r=np.asarray(data['residual']);m=np.asarray(data['monodromy']);step=np.linalg.solve(m-np.eye(len(x)),-r)
        correction=float(max(abs(step))/max(1.,max(abs(x))))
        row={'iteration':iteration,'state':x.tolist(),'residual_inf':float(max(abs(r))),'step':step.tolist(),'relative_correction':correction,'trials':[]}
        history.append(row)
        if correction<=1e-10:
            x=x+step;last=evaluate(x,s,period);qualification=qualify(last)
            return {'converged':True,'qualified':qualification['qualified'],'state':x.tolist(),'history':history,'data':last,'qualification':qualification}
        for half in range(13):
            alpha=2.**-half;trial=x+alpha*step;tr=evaluate(trial,s,period)
            merit=float(max(abs(np.asarray(tr['residual']))));accepted=merit<float(max(abs(r)))
            row['trials'].append({'alpha':alpha,'state':trial.tolist(),'residual_inf':merit,'accepted':accepted})
            if accepted:x=trial;break
        else:return {'converged':False,'qualified':False,'reason':'line_search_stalled','state':x.tolist(),'history':history,'data':last}
    return {'converged':False,'qualified':False,'reason':'iterations_exhausted','state':x.tolist(),'history':history,'data':last}


def refine_flip(evaluate,x,s,period,emit=lambda *_:None):
    data=evaluate(x,s,period);m=np.asarray(data['monodromy']);values,vectors,_=spectrum(m)
    i=int(np.argmin(abs(values+1)));q=np.real(vectors[:,i]);q/=np.linalg.norm(q)
    if q[np.argmax(abs(q))]<0:q=-q
    n=len(x);calls=[]
    def augmented(z):
        d=evaluate(z[:n],z[n],period);mm=np.asarray(d['monodromy']);v=z[n+1:]
        g=np.r_[d['residual'],(mm+np.eye(n))@v,(v@v-1)/2]
        row={'z':z.tolist(),'g':g.tolist(),'evaluation_id':d.get('id')};calls.append(row);emit('flip_augmented',row)
        return g
    solved=root(augmented,np.r_[x,s,q],method='hybr',options={'xtol':1e-10,'maxfev':500})
    final=augmented(solved.x);xx,ss,qq=solved.x[:n],float(solved.x[n]),solved.x[n+1:]
    data=evaluate(xx,ss,period);qual=qualify(data);m=np.asarray(data['monodromy']);vals,vecs,res=spectrum(m)
    j=int(np.argmin(abs(vals+1)));lv,lvec=np.linalg.eig(m.T);k=int(np.argmin(abs(lv+1)));left=np.real(lvec[:,k]);overlap=float(left@qq)
    if abs(overlap)<=1e-8:raise ValueError('ill-conditioned left/right pairing')
    left/=overlap;others=np.delete(vals,j)
    gates={'solver_success':bool(solved.success),'augmented_residual':float(max(abs(final)))<1e-8,
           'primitive_orbit':qual['qualified'],'minus_one':abs(vals[j]+1)<1e-6,
           'simple_minus_one':float(min(abs(others+1)))>1e-4,
           'other_modes_attracting':float(max(abs(others)))<1-1e-7,
           'left_residual':float(np.linalg.norm(m.T@left+left,np.inf)/max(1.,np.linalg.norm(m,np.inf)*np.linalg.norm(left,np.inf)))<1e-9}
    return {'qualified':all(gates.values()),'gates':gates,'state':xx.tolist(),'s':ss,'q':qq.tolist(),'left':left.tolist(),
            'nfev':int(solved.nfev),'solver_message':str(solved.message),'augmented_residual_inf':float(max(abs(final))),
            'calls':calls,'data':data,'qualification':qual,'other_radius':float(max(abs(others)))}


def crossing(evaluate,point,period,emit=lambda *_:None):
    x=np.asarray(point['state']);s=point['s'];q=np.asarray(point['q']);left=np.asarray(point['left']);m=np.asarray(point['data']['monodromy'])
    b=np.asarray(point['data']['parameter_derivative']);tangent=np.linalg.solve(m-np.eye(len(x)),-b)
    rows=[]
    for exponent in (12,14,16,18):
        eps=2.**-exponent;nodes=[];eigs=[]
        for sign in (-1,1):
            node=correct(evaluate,x+sign*eps*tangent,s+sign*eps,period);nodes.append(node)
            if not node['qualified']:raise ValueError('crossing branch correction failed')
            vals=np.array([complex(*z) for z in node['qualification']['multipliers']]);eigs.append(vals[np.argmin(abs(vals+1))])
        mp=np.asarray(evaluate(x+eps*tangent,s+eps,period)['monodromy']);mm=np.asarray(evaluate(x-eps*tangent,s-eps,period)['monodromy'])
        total=float(left@((mp-mm)/(2*eps))@q)
        slope=float((eigs[1].real-eigs[0].real)/(2*eps))
        row={'exponent':exponent,'epsilon':eps,'nodes':nodes,'total_tangent_crossing':total,'corrected_branch_crossing':slope,
             'real_pair':bool(max(abs(z.imag) for z in eigs)<1e-8)};rows.append(row);emit('crossing',row)
    a,b=rows[-2:];u,v=a['corrected_branch_crossing'],b['corrected_branch_crossing']
    gates={'nonzero_same_sign':u*v>0 and abs(v)>1e-6,'step_convergence':abs(u-v)/max(abs(u),abs(v))<1e-3,
           'total_derivative_agreement':abs(v-b['total_tangent_crossing'])/max(abs(v),1e-12)<1e-3,'real_pair':all(r['real_pair'] for r in rows)}
    return {'qualified':all(gates.values()),'gates':gates,'tangent':tangent.tolist(),'series':rows}


def doubled(evaluate,point,period,amplitude,seed=None,emit=lambda *_:None):
    x=np.asarray(point['state']);s=point['s'];q=np.asarray(point['q']);left=np.asarray(point['left']);n=len(x)
    z0=np.r_[x+amplitude*q,s] if seed is None else np.asarray(seed)
    cache={};calls=[]
    def pair(z):
        key=z.tobytes()
        if cache.get('key')==key:return cache['g'],cache['jac']
        d=evaluate(z[:n],z[n],2*period);m=np.asarray(d['monodromy']);b=np.asarray(d['parameter_derivative'])
        # Returned middle phase uses exactly the same forward orbit.
        middle=d['checkpoints'][str(period)];u=np.asarray(middle['state']);m1=np.asarray(middle['monodromy']);b1=np.asarray(middle['parameter_derivative'])
        amp=float(left@(z[:n]-u)/2)
        g=np.r_[d['residual'],amp-amplitude]
        jac=np.vstack([np.column_stack([m-np.eye(n),b]),np.r_[left@(np.eye(n)-m1)/2,-left@b1/2]])
        row={'z':z.tolist(),'g':g.tolist(),'amplitude':amp,'evaluation_id':d.get('id')};calls.append(row);emit('doubled_augmented',row)
        cache.update(key=key,g=g,jac=jac);return g,jac
    solved=root(lambda z:pair(z)[0],z0,jac=lambda z:pair(z)[1],method='hybr',options={'xtol':1e-10,'maxfev':220})
    g,_=pair(solved.x);d=evaluate(solved.x[:n],solved.x[n],2*period);qual=qualify(d)
    parent=correct(evaluate,x,float(solved.x[n]),period)
    scale=max(1.,max(abs(x)));separation=float(max(abs(np.asarray(d['orbit'][period])-np.asarray(d['orbit'][0])))/scale)
    distance=float(max(abs(solved.x[:n]-x))/scale)
    gates={'solver_success':bool(solved.success),'augmented_residual':float(max(abs(g)))<1e-8,
        'primitive_doubled':qual['qualified'],'noncollapsed':separation>=1e-6,
        'local_to_flip':distance<.05 and abs(float(solved.x[n])-s)<.01,'parent_qualified':parent['qualified']}
    return {'qualified':all(gates.values()),'gates':gates,'amplitude_requested':float(amplitude),'state':solved.x[:n].tolist(),
        's':float(solved.x[n]),'delta_s':float(solved.x[n]-s),'phase_separation':separation,'distance_to_flip':distance,
        'augmented_residual_inf':float(max(abs(g))),'nfev':int(solved.nfev),'solver_message':str(solved.message),
        'data':d,'qualification':qual,'parent':parent,'calls':calls}
