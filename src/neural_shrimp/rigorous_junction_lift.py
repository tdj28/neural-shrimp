"""Conditional normalized full-native lift of an already proved unit/flip root.

No root is discovered here. The caller authenticates the full 4n+3 root,
center value and whole original Jacobian of the SAME model and box. All
subsequent native phases are freshly acquired on their complete propagated
state/mode/rate enclosures; no old derivative domain is silently enlarged.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
from neural_shrimp.rigorous_eigenpair_locus import require,finite,column
from neural_shrimp.rigorous_orbits import eye,ball_json,vector_json,matrix_json,norm_inf_upper
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball
from neural_shrimp.rigorous_unit_pair import phase,terminal_matrix


def endpoints(v):
    if isinstance(v,arb):v=ball_json(v)
    require(set(v)>= {'mid','rad'},'finite serialized original domain')
    m,e=v['mid'];r,f=v['rad'];require(type(m)is type(e)is type(r)is type(f)is int and r>=0,'exact finite dyadic ball')
    mid=F(m)*F(2)**e;rad=F(r)*F(2)**f;return mid-rad,mid+rad


def contained(inner,outer):
    a,b=endpoints(inner);c,d=endpoints(outer);return c<=a<=b<=d


def sharpen_known_root(center,whole,C,value,J,*,iterations=3,tick=lambda:None,observe=lambda k,v:None):
    """Intersect the existing root with e=-C H(c)+(I-C A)e on original box.

    H(c) and J must already be authenticated acquisitions. A is the mean
    value along c to the known root, within the ORIGINAL literal box. This
    routine is conditional on that root, never an alternative existence test.
    """
    n=len(center);require(1<=n<=55 and len(whole)==n,'full known-root size')
    require(type(iterations)is int and 0<=iterations<=3,'fixed intersection count')
    require(all(x.is_finite()and x.is_exact()for x in center),'exact known center')
    require(C.nrows()==C.ncols()==J.nrows()==J.ncols()==value.nrows()==n and value.ncols()==1,'complete same-box arrays')
    require(all(finite(a)for a in(C,value,J))and all(x.is_exact()for row in C.tolist()for x in row),'finite exact inverse proposal and interval derivatives')
    original=[original_ball(v)for v in whole]
    require(all(contained(c,b)for c,b in zip(center,whole,strict=True)),'center in original literal derivative box')
    tick();D=eye(n)-C*J;q=norm_inf_upper(D);require(q.is_finite()and 0<=q<1,'strict complete known-root defect')
    forcing=-C*value;error=column([b-c for b,c in zip(original,center,strict=True)])
    observe('known_root_refinement_inputs',dict(center=vector_json(center),whole=whole,preconditioner=matrix_json(C),value=matrix_json(value),whole_jacobian=matrix_json(J),defect=matrix_json(D),contraction=ball_json(q)))
    rows=[]
    for k in range(iterations):
        image=forcing+D*error;require(finite(image),'finite fixed-point image')
        error=arb_mat([[error[i,0].intersection(image[i,0])]for i in range(n)])
        require(finite(error),'nonempty known-root intersection')
        row=dict(iteration=k,image=matrix_json(image),retained=matrix_json(error));observe('known_root_intersection',row);rows.append(row);tick()
    box=[center[i]+error[i,0]for i in range(n)]
    require(all(contained(x,b)for x,b in zip(box,whole,strict=True)),'refined full root stays in literal original derivative box')
    return box,dict(passed=True,dimension=n,box=vector_json(box),contraction=ball_json(q),intersections=rows,scope='Conditional sharpening of the same authenticated existing root; no Newton iteration or new existence proof.')


def lift_known_junction(model,box,period,*,tick=lambda:None,observe=lambda k,v:None):
    """Pack (x_i,q_i,r_i) for all native phases, both rates and t=angle².

    Input order is x, rates, q, r, t, w for the proved unit-pair/flip root.
    w is discarded only after its proved local junction authority is bound by
    the caller. Phase/sign/gauge changes are prohibited. Two complete native
    MixedJet evaluations per phase, each preceded by its exact input record.
    """
    n=model.n;require(type(n)is int and 2<=n<=13 and type(period)is int and 1<=period<=18,'bounded native dimensions')
    require(len(box)==4*n+3 and all(v.is_finite()for v in box),'complete same junction root')
    state=list(box[:n]);rates=list(box[n:n+2]);q=list(box[n+2:2*n+2]);r=list(box[2*n+2:3*n+2]);t=box[3*n+2]
    require(all(v>0 for v in rates)and 0<t and t<=arb(1)/4,'positive actual rates and bounded nonzero angle')
    flat=[];returns=[]
    class Recorded:
        def __init__(self):self.n=n;self.calls=0
        def cusp_step(self,x,h,o):
            phase_index=self.calls//2;mode=('q','r')[self.calls%2]
            # MixedJet fields give exact acquired base/direction/rate domains.
            observe('lift_native_input',dict(phase=phase_index,direction=mode,state=vector_json([v.f for v in x]),mode=vector_json([v.d for v in x]),rates=vector_json([h.f,o.f]),base_gradients=[vector_json(v.g)for v in x+[h,o]],mixed=[vector_json(v.m)for v in x+[h,o]]))
            self.calls+=1;return model.cusp_step(x,h,o)
    recorded=Recorded()
    for k in range(period):
        flat.extend(state+q+r)
        def retain(kind,row):
            observe(kind,row);returns.append(row)
        v,_,_=phase(recorded,state,q,r,rates,tick,retain,k)
        state=[v[i,0]for i in range(n)];q=[v[n+i,0]for i in range(n)];r=[v[2*n+i,0]for i in range(n)]
    require(recorded.calls==2*period and len(returns)==2*period,'complete literal native lift')
    flat.extend(rates+[t]);difference=column(state+q+r)-terminal_matrix(n,t)*column(flat[:3*n])
    norm=sum((v*v for v in box[n+2:2*n+2]),arb(0))-1
    orth=sum((a*b for a,b in zip(box[n+2:2*n+2],box[2*n+2:3*n+2],strict=True)),arb(0))
    return flat,dict(dimension=n,period=period,shooting_dimension=len(flat),box=vector_json(flat),closure_residual=matrix_json(difference),closure_contains_zero=all(difference[i,0].contains(0)for i in range(3*n)),normalization_contains_zero=norm.contains(0)and orth.contains(0),native_calls=recorded.calls,scope='Conditional native lift of the same independently proved normalized junction root. Closure zero containment alone is not a root proof; no continuation membership or complement extension is asserted.')
