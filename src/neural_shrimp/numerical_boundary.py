"""Numerical two-rate boundary continuation using analytic mixed jets.

Arb supplies derivative midpoints; no interval existence or geometry proof is
claimed by this module. Historical rigorous kernels are unchanged.
"""
from __future__ import annotations
import numpy as np
from flint import arb, ctx
from .rigorous_folds import FoldTanhMap, MixedJet
from .rigorous_orbits import exact, dyadic


def coordinate_ball(x):
    """Preserve each finite native coordinate's exact dyadic value."""
    n,d=x.as_integer_ratio()
    return exact(f'{n}/{d}')


class TwoRateBoundary:
    """Unknowns (theta, actual hidden rate, actual output rate, q).

    Residual (F^p(theta)-theta, (D F^p-sigma I)q, (q.q-1)/2).
    This has 2n+1 equations in 2n+2 variables for sigma in {-1,+1}.
    """
    def __init__(self, inputs, period, sigma, precision=128):
        if sigma not in (-1,1) or period<1: raise ValueError('invalid boundary')
        self.model=FoldTanhMap(inputs);self.period=period;self.sigma=sigma;self.precision=precision
        self.n=self.model.n

    def __call__(self,z):
        z=np.asarray(z,dtype=np.longdouble);n=self.n
        if z.shape!=(2*n+2,) or not np.isfinite(z).all() or min(z[n:n+2])<=0:
            raise ValueError('invalid two-rate boundary state')
        ctx.prec=self.precision
        theta=[coordinate_ball(x) for x in z[:n]];q=[coordinate_ball(x) for x in z[n+2:]]
        current=[MixedJet.variable(theta[i],n+2,i,q[i]) for i in range(n)]
        hr=MixedJet.variable(coordinate_ball(z[n]),n+2,n)
        out=MixedJet.variable(coordinate_ball(z[n+1]),n+2,n+1)
        previous_rates=self.model.rates
        self.model.rates=[hr]*self.model.hidden+[out]*(n-self.model.hidden)
        orbit=[list(theta)]
        try:
            for _ in range(self.period):
                current=self.model.jet_step(current,hr);orbit.append([j.f for j in current])
        finally:self.model.rates=previous_rates
        residual=[current[i].f-theta[i] for i in range(n)]
        mono=[[current[i].g[j] for j in range(n)] for i in range(n)]
        first=[[current[i].g[j]-int(i==j) for j in range(n)]+list(current[i].g[n:])+[arb(0)]*n for i in range(n)]
        second=[list(current[i].m)+[mono[i][j]-self.sigma*int(i==j) for j in range(n)] for i in range(n)]
        value=residual+[current[i].d-self.sigma*q[i] for i in range(n)]+[(sum((v*v for v in q),arb(0))-1)/2]
        deriv=first+second+[[arb(0)]*(n+2)+q]
        def floats(x):
            ans=np.asarray([[float(v.mid()) for v in row] for row in x]);
            if not np.isfinite(ans).all():raise FloatingPointError('nonfinite derivative midpoint')
            return ans
        return {'value':floats([value])[0],'jacobian':floats(deriv),'orbit':floats(orbit),'monodromy':floats(mono)}


def null_tangent(jacobian,reference=None):
    """Numerical simple-nullspace direction; ambiguity is an explicit failure."""
    a=np.asarray(jacobian,dtype=float)
    if a.ndim!=2 or a.shape[0]<1 or not np.isfinite(a).all():raise ValueError('invalid tangent matrix')
    u,s,vh=np.linalg.svd(a,full_matrices=True)
    if a.shape[1]!=a.shape[0]+1 or not np.isfinite(s).all() or s[-1]<1e-10*max(1,s[0]):
        raise ValueError('boundary row rank unresolved')
    t=vh[-1];res=np.linalg.norm(a@t)/max(1,np.linalg.norm(a))
    if res>1e-11:raise ValueError('tangent residual unresolved')
    if reference is not None:
        cosine=float(t@reference)
        if abs(cosine)<1e-6:raise ValueError('tangent orientation unresolved')
        if cosine<0:t=-t
    return t,s


def correct_section(evaluate,prediction,normal,scale,max_iterations=10,tolerance=2e-11,emit=lambda kind,row:None):
    """Newton corrector on a fixed scaled pseudoarclength plane.

    Every trial evaluation is passed through the caller's durable journal.
    Arbitrary sign changes of q or optimizer success cannot pass the residual gate.
    """
    pred=np.asarray(prediction,dtype=np.longdouble);scale=np.asarray(scale,dtype=float)
    if pred.shape!=scale.shape or not np.all(scale>0):raise ValueError('invalid scales')
    normal=np.asarray(normal,dtype=float)
    if normal.shape!=pred.shape or abs(np.linalg.norm(normal)-1)>1e-10:raise ValueError('invalid plane normal')
    z=pred.copy();history=[]
    def assemble(point):
        data=evaluate(point);plane=float(normal@((point-pred)/scale))
        v=np.r_[np.asarray(data['value'],dtype=float),plane]
        return data,v,np.vstack([np.asarray(data['jacobian'],dtype=float)*scale,normal])
    for iteration in range(max_iterations):
        data,v,j=assemble(z);err=float(np.max(np.abs(v)));history.append(err)
        emit('newton',{'iteration':iteration,'point':z,'error':err,'plane':float(v[-1])})
        if err<tolerance:return {'qualified':True,'state':z,'data':data,'history':history}
        delta=np.linalg.solve(j,-v)
        if not np.isfinite(delta).all():raise FloatingPointError('nonfinite correction')
        accepted=False
        for half in range(12):
            trial=z+scale*delta*2.**(-half)
            try:trialdata,trialv,_=assemble(trial)
            except (ValueError,FloatingPointError) as error:
                emit('invalid_trial',{'point':trial,'error':str(error)});continue
            trialerr=float(np.max(np.abs(trialv)))
            emit('line_search',{'iteration':iteration,'half':half,'point':trial,'error':trialerr})
            if trialerr<err:
                z=trial;accepted=True;break
        if not accepted:return {'qualified':False,'state':z,'history':history,'reason':'line_search_failed'}
    return {'qualified':False,'state':z,'history':history,'reason':'iteration_limit','last_trial_error':trialerr,'last_trial_data':trialdata}


def qualify_boundary(data,sigma,period):
    """Floating full-state primitive and simple-critical-multiplier diagnostics."""
    orbit=data['orbit'];m=data['monodromy'];scale=max(1,float(np.max(abs(orbit))))
    eig,vec=np.linalg.eig(m);res=np.max(np.linalg.norm(m@vec-vec*eig,axis=0)/np.maximum(1,np.linalg.norm(m,ord=np.inf)*np.linalg.norm(vec,axis=0)))
    critical=int(np.argmin(abs(eig-sigma)));others=np.delete(eig,critical)
    divisions={str(d):float(np.max(abs(orbit[d]-orbit[0]))/scale) for d in range(1,period) if period%d==0}
    good=bool(np.max(abs(data['value']))<2e-10 and abs(eig[critical]-sigma)<1e-7 and (len(others)==0 or min(abs(others-sigma))>1e-5) and res<1e-10 and all(v>1e-6 for v in divisions.values()))
    return {'qualified':good,'critical_index':critical,'multipliers_real':eig.real.tolist(),'multipliers_imag':eig.imag.tolist(),'eigen_residual':float(res),'proper_divisor_separations':divisions,'complement_radius':float(max(abs(others))) if len(others) else 0.,'closure':float(np.max(abs(orbit[-1]-orbit[0]))/scale)}
