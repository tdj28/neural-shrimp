"""Exact generalized-chain root preparation for a possible double +1 event.

R=F^p(x;a,b)-x and A=D_x R. The equations are R=0, Aq=0,
Ar=q, (q.q-1)/2=0, q.r=0. This is not the cusp system. A root
proves a nonzero length-two Jordan chain, not exact algebraic multiplicity,
a generic 1:1 resonance, attraction, a branch connection, or a shrimp.
"""
from flint import arb, arb_mat
from .rigorous_folds import MixedJet
from .rigorous_orbits import eye, midpoint_matrix, norm_inf_upper, ball_json, vector_json, matrix_json


def augmented_chain(model, z, period, tick=lambda: None):
    """Return value and full Jacobian at z=(x,a,b,q,r), using two mixed passes.

    Each pass differentiates with respect to x,a,b while its direction is held
    constant. The exact A and -I blocks restore derivatives in q and r.
    """
    n=model.n;dim=3*n+2;base=n+2
    if type(n)is not int or n<1 or len(z)!=dim or type(period)is not int or not 1<=period<=64:
        raise ValueError('generalized-chain dimensions or period')
    x=z[:n];q=z[base:base+n];r=z[base+n:]
    def propagate(direction):
        state=[MixedJet.variable(x[i],base,i,direction[i])for i in range(n)]
        a=MixedJet.variable(z[n],base,n);b=MixedJet.variable(z[n+1],base,n+1)
        initial=state;orbit=[[v.f for v in state]]
        for _ in range(period):
            tick();state=model.cusp_step(state,a,b);orbit.append([v.f for v in state])
        return [u-v for u,v in zip(state,initial)],orbit
    first,orbit=propagate(q);second,_=propagate(r)
    a=arb_mat([[v.g[k]for k in range(n)]for v in first])
    value=[v.f for v in first]+[v.d for v in first]+[v.d-q[i]for i,v in enumerate(second)]
    value += [(sum((v*v for v in q),arb(0))-1)/2,sum((q[i]*r[i]for i in range(n)),arb(0))]
    jac=[list(v.g)+[arb(0)]*(2*n)for v in first]
    jac += [list(v.m)+[a[i,k]for k in range(n)]+[arb(0)]*n for i,v in enumerate(first)]
    jac += [list(v.m)+[arb(-int(i==k))for k in range(n)]+[a[i,k]for k in range(n)]for i,v in enumerate(second)]
    jac += [[arb(0)]*base+list(q)+[arb(0)]*n,[arb(0)]*base+list(r)+list(q)]
    return arb_mat([[v]for v in value]),arb_mat(jac),{'orbit':orbit,'a':a,'monodromy':a+eye(n),'q':q,'r':r}


def chain_root_proof(model, initial, period, *,radius_exponent=-70,newton_steps=0,tick=lambda:None,observe=lambda kind,record:None):
    """Outward-rounded root enclosure only; no codimension-two classification."""
    n=model.n;dim=3*n+2
    if len(initial)!=dim or type(newton_steps)is not int or not 0<=newton_steps<=10 or type(radius_exponent)is not int or not -256<=radius_exponent<=-1:
        raise ValueError('generalized-chain proof settings')
    center=[v.mid()for v in initial]
    for iteration in range(newton_steps):
        v,j,_=augmented_chain(model,center,period,tick)
        observe('newton_evaluation',{'iteration':iteration,'center':vector_json(center),'value':matrix_json(v),'jacobian':matrix_json(j)})
        step=j.inv()*v;center=[(center[i]-step[i,0]).mid()for i in range(dim)]
        observe('newton_proposal',{'iteration':iteration,'center':vector_json(center)})
    value,j,_=augmented_chain(model,center,period,tick);pre=midpoint_matrix(j.inv())
    observe('center_evaluation',{'center':vector_json(center),'value':matrix_json(value),'jacobian':matrix_json(j)})
    radius=arb((1,radius_exponent));box=[arb(v,radius)for v in center]
    _,whole,data=augmented_chain(model,box,period,tick)
    defect=eye(dim)-pre*whole;bound=norm_inf_upper(defect)
    image=-pre*value+defect*arb_mat([[arb(0,radius)]for _ in center])
    inclusions=[bool(box[i].contains_interior(center[i]+image[i,0]))for i in range(dim)]
    root=bool(all(inclusions) and bound.is_finite() and bound<1)
    divisors=[]
    for d in range(1,period):
        if period%d:continue
        differences=[data['orbit'][d][i]-box[i]for i in range(n)]
        divisors.append({'divisor':d,'excluded':any(v.is_finite() and not v.contains(0)for v in differences),'differences':vector_json(differences)})
    gates={'normalized_augmented_root_unique':root,'positive_actual_rates':bool(box[n]>0 and box[n+1]>0),'primitive_period':all(v['excluded']for v in divisors)}
    result={'schema':'neural-shrimp-generalized-chain-root-v1','passed':all(gates.values()),'gates':gates,'period':period,'dimension':n,'augmented_dimension':dim,'settings':{'radius_exponent':radius_exponent,'newton_steps':newton_steps},'center':vector_json(center),'box':vector_json(box),'value':matrix_json(value),'preconditioner':matrix_json(pre),'whole_jacobian':matrix_json(whole),'defect':matrix_json(defect),'image_correction':matrix_json(image),'coordinate_inclusions':inclusions,'contraction':ball_json(bound),'monodromy':matrix_json(data['monodromy']),'orbit_enclosures':[vector_json(v)for v in data['orbit']],'divisors':divisors,'statement':'Locally unique normalized primitive periodic-orbit generalized-chain root only; exact multiplicity, generic 1:1 resonance, complementary spectrum, attraction, connected boundary and shrimp organization are not established.'}
    observe('chain_root_gate',result);return result
