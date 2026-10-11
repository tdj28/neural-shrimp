"""Exact mixed jets for two-parameter cusps of a discrete periodic-orbit map.

The augmented root is (R, R_x q, R_x h+R_xx[q,q], (q.q-1)/2, q.h).
A root alone is not a generic dynamical cusp: separate cubic, unfolding,
primitive-period and noncritical-unit-circle gates remain necessary.
"""
from flint import arb,arb_mat
from .rigorous_folds import FoldTanhMap

class CuspJet:
    """Value and two ordinary path derivatives, each with full base gradient.

    These are derivatives, not factorial-scaled Taylor coefficients. Initial
    path x(t)=theta+t*q+t²*h/2 includes independent base variables theta,q,h.
    """
    __slots__=('f','g','d','m','d2','m2')
    def __init__(self,f,g,d,m,d2,m2):
        self.f,self.g,self.d,self.m,self.d2,self.m2=f,tuple(g),d,tuple(m),d2,tuple(m2)
    @classmethod
    def variable(cls,value,size,index,direction=0,direction_index=None,acceleration=0,acceleration_index=None):
        return cls(arb(value),[arb(i==index) for i in range(size)],arb(direction),
                   [arb(i==direction_index) for i in range(size)],arb(acceleration),
                   [arb(i==acceleration_index) for i in range(size)])
    def __add__(self,b):
        if not isinstance(b,CuspJet):return CuspJet(self.f+b,self.g,self.d,self.m,self.d2,self.m2)
        if len(self.g)!=len(b.g):raise ValueError('jet dimensions')
        return CuspJet(self.f+b.f,[x+y for x,y in zip(self.g,b.g)],self.d+b.d,
                       [x+y for x,y in zip(self.m,b.m)],self.d2+b.d2,[x+y for x,y in zip(self.m2,b.m2)])
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,b):return self+(-b)
    def __rsub__(self,b):return -self+b
    def __mul__(self,b):
        if not isinstance(b,CuspJet):return CuspJet(self.f*b,[x*b for x in self.g],self.d*b,[x*b for x in self.m],self.d2*b,[x*b for x in self.m2])
        if len(self.g)!=len(b.g):raise ValueError('jet dimensions')
        return CuspJet(self.f*b.f,[x*b.f+self.f*y for x,y in zip(self.g,b.g)],
            self.d*b.f+self.f*b.d,
            [x*b.f+self.d*y+z*b.d+self.f*w for x,y,z,w in zip(self.m,b.g,self.g,b.m)],
            self.d2*b.f+2*self.d*b.d+self.f*b.d2,
            [x*b.f+self.d2*y+2*(z*b.d+self.d*w)+u*b.d2+self.f*v for x,y,z,w,u,v in zip(self.m2,b.g,self.m,b.m,self.g,b.m2)])
    __rmul__=__mul__
    def tanh(self):
        t=self.f.tanh();s=1-t*t;c=-2*t*s;third=-2*s*s+4*t*t*s
        return CuspJet(t,[s*x for x in self.g],s*self.d,
            [s*m+c*g*self.d for m,g in zip(self.m,self.g)],s*self.d2+c*self.d*self.d,
            [s*m2+c*g*self.d2+2*c*self.d*m+third*g*self.d*self.d for m2,g,m in zip(self.m2,self.g,self.m)])

class TwoRateCuspMap(FoldTanhMap):
    def cusp_step(self,state,hidden,output):
        previous=self.rates
        self.rates=[hidden]*self.hidden+[output]*(self.n-self.hidden)
        try:return self.jet_step(state,hidden)
        finally:self.rates=previous

def augmented_cusp(model,z,period,tick=lambda:None):
    n=model.n;dim=3*n+2
    if len(z)!=dim or type(period) is not int or not 1<=period<=64:raise ValueError('augmented dimensions/period')
    state=[CuspJet.variable(z[i],dim,i,z[n+2+i],n+2+i,z[2*n+2+i],2*n+2+i) for i in range(n)]
    hidden=CuspJet.variable(z[n],dim,n);output=CuspJet.variable(z[n+1],dim,n+1)
    orbit=[[v.f for v in state]]
    for _ in range(period):tick();state=model.cusp_step(state,hidden,output);orbit.append([v.f for v in state])
    theta=z[:n];q=z[n+2:2*n+2];h=z[2*n+2:]
    value=[v.f-theta[i] for i,v in enumerate(state)]+[v.d-q[i] for i,v in enumerate(state)]+[v.d2-h[i] for i,v in enumerate(state)]+[(sum((v*v for v in q),arb(0))-1)/2,sum((q[i]*h[i] for i in range(n)),arb(0))]
    jac=[[v.g[j]-int(i==j) for j in range(dim)] for i,v in enumerate(state)]
    jac += [[v.m[j]-int(n+2+i==j) for j in range(dim)] for i,v in enumerate(state)]
    jac += [[v.m2[j]-int(2*n+2+i==j) for j in range(dim)] for i,v in enumerate(state)]
    jac += [[arb(0)]*(n+2)+list(q)+[arb(0)]*n,[arb(0)]*(n+2)+list(h)+list(q)]
    return arb_mat([[v] for v in value]),arb_mat(jac),{'orbit':orbit,'monodromy':arb_mat([list(v.g[:n]) for v in state]),
        'base_gradients':arb_mat([list(v.g) for v in state]),'direction_gradients':arb_mat([list(v.m) for v in state]),
        'acceleration_gradients':arb_mat([list(v.m2) for v in state]),'first_direction':[v.d for v in state],'second_path_derivative':[v.d2 for v in state]}


def spectral_exclusion(matrix):
    """Validated similarity and Gershgorin exclusion of the entire unit circle.

    Floating eigenvectors are only an exact-dyadic basis proposal. An Arb/Acb
    inverse and outward-rounded transformed matrix/discs supply every gate.
    The matrix here is M-q*w^T, so its spectrum is zero plus the complement.
    """
    import numpy as np
    from flint import acb,acb_mat
    from .rigorous_orbits import exact,dyadic,ball_json
    n=matrix.nrows();proposal=np.array([[float(matrix[i,j].mid()) for j in range(n)] for i in range(n)])
    _,vec=np.linalg.eig(proposal)
    if not np.isfinite(vec).all():raise ValueError('nonfinite basis proposal')
    basis=acb_mat([[acb(exact(dyadic(vec[i,j].real)),exact(dyadic(vec[i,j].imag))) for j in range(n)] for i in range(n)])
    inverse=basis.inv();transformed=inverse*acb_mat([[acb(matrix[i,j]) for j in range(n)] for i in range(n)])*basis
    discs=[]
    for i in range(n):
        z=transformed[i,i];center=acb(z.real.mid(),z.imag.mid())
        radius=(abs(z-center)+sum((abs(transformed[i,j]) for j in range(n) if j!=i),arb(0))).upper()
        modulus=abs(center);inside=bool(modulus+radius<1);outside=bool(modulus-radius>1)
        discs.append({'center_real':ball_json(center.real),'center_imag':ball_json(center.imag),'radius':ball_json(radius),'strict_inside':inside,'strict_outside':outside,'unit_circle_excluded':inside or outside})
    def encode(m):return [[{'real':ball_json(m[i,j].real),'imag':ball_json(m[i,j].imag)} for j in range(n)] for i in range(n)]
    return {'passed':all(d['unit_circle_excluded'] for d in discs),'proposal_basis':encode(basis),'validated_inverse':encode(inverse),'transformed_matrix':encode(transformed),'discs':discs,'scope':'Entire spectrum avoids unit circle; disc counts alone do not count unstable eigenvalues when discs overlap.'}


def cusp_proof(model,initial,period,*,radius_exponent=-80,newton_steps=8,tick=lambda:None,observe=lambda k,r:None):
    from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,vector_json,matrix_json
    n=model.n;dim=3*n+2
    if len(initial)!=dim or type(newton_steps) is not int or not 0<=newton_steps<=10 or not -256<=radius_exponent<=-1:raise ValueError('proof settings')
    center=[v.mid() for v in initial]
    for iteration in range(newton_steps):
        v,j,data=augmented_cusp(model,center,period,tick)
        observe('newton_evaluation',{'iteration':iteration,'center':vector_json(center),'residual':matrix_json(v),'jacobian':matrix_json(j)})
        step=j.inv()*v
        center=[(center[i]-step[i,0]).mid() for i in range(dim)]
        observe('newton_proposal',{'iteration':iteration,'center':vector_json(center)})
    v,j,data0=augmented_cusp(model,center,period,tick)
    observe('center_evaluation',{'center':vector_json(center),'residual':matrix_json(v),'jacobian':matrix_json(j)})
    pre=midpoint_matrix(j.inv());radius=arb((1,radius_exponent));box=[arb(c,radius) for c in center]
    _,whole,data=augmented_cusp(model,box,period,tick)
    defect=eye(dim)-pre*whole;norm=norm_inf_upper(defect);image=-pre*v+defect*arb_mat([[arb(0,radius)] for _ in box])
    inclusion=[bool(box[i].contains_interior(center[i]+image[i,0])) for i in range(dim)]
    root=bool(all(inclusion) and norm.is_finite() and norm<1)
    observe('root_gate',{'passed':root,'coordinate_inclusions':inclusion,'contraction':ball_json(norm),'whole_jacobian':matrix_json(whole)})
    result={'schema':'neural-shrimp-cusp-v1','passed':False,'settings':{'radius_exponent':radius_exponent,'newton_steps':newton_steps},'center':vector_json(center),'box':vector_json(box),'residual':matrix_json(v),'preconditioner':matrix_json(pre),'whole_jacobian':matrix_json(whole),'defect':matrix_json(defect),'image_correction':matrix_json(image),'coordinate_inclusions':inclusion,'contraction':ball_json(norm),'gates':{'augmented_root':root,'positive_rates':bool(box[n]>0 and box[n+1]>0)},'orbit_enclosures':[vector_json(row) for row in data['orbit']]}
    if not root:return result
    m=data['monodromy'];a=m-eye(n);q=arb_mat([[x] for x in box[n+2:2*n+2]]);h=arb_mat([[x] for x in box[2*n+2:]])
    border=arb_mat([[a[i,j] if i<n and j<n else q[i,0] if i<n else q[j,0] if j<n else arb(0) for j in range(n+1)] for i in range(n+1)])
    inv=border.inv();rhs=arb_mat([[arb(int(i==n))] for i in range(n+1)]);left_aug=inv.transpose()*rhs;left=arb_mat([[left_aug[i,0] for i in range(n)]])
    grad=data['base_gradients'];g1=data['direction_gradients'];g2=data['acceleration_gradients']
    bqh=arb_mat([[sum((g1[i,j]*h[j,0] for j in range(n)),arb(0))] for i in range(n)])
    third=arb_mat([[sum((g2[i,j]*q[j,0] for j in range(n)),arb(0))+2*bqh[i,0]] for i in range(n)])
    cubic=(left*third)[0,0]/6
    unfold=arb_mat(2,2);parameter_solutions=[]
    for col in range(2):
        rlambda=arb_mat([[grad[i,n+col]] for i in range(n)])
        y=inv*arb_mat([[-rlambda[i,0]] for i in range(n)]+[[arb(0)]])
        unfold[0,col]=(left*rlambda)[0,0]
        unfold[1,col]=sum((left[0,i]*(g1[i,n+col]+sum((g1[i,j]*y[j,0] for j in range(n)),arb(0))) for i in range(n)),arb(0))
        parameter_solutions.append(matrix_json(y))
    determinant=unfold.det();deflated=m-q*left;spectral=spectral_exclusion(deflated)
    divisors=[]
    for d in range(1,period):
        if period%d:continue
        differences=[data['orbit'][d][i]-box[i] for i in range(n)]
        divisors.append({'divisor':d,'excluded':any(v.is_finite() and not v.contains(0) for v in differences),'differences':vector_json(differences)})
    finite_border=all(inv[i,j].is_finite() for i in range(n+1) for j in range(n+1))
    result['gates'].update(simple_plus_one=finite_border,cubic_nonzero=bool(cubic.is_finite() and not cubic.contains(0)),versal_unfolding=bool(determinant.is_finite() and not determinant.contains(0)),complement_unit_circle_excluded=spectral['passed'],primitive_period=all(d['excluded'] for d in divisors))
    result.update(passed=all(result['gates'].values()),monodromy=matrix_json(m),border=matrix_json(border),validated_border_inverse=matrix_json(inv),left_augmented=matrix_json(left_aug),left=matrix_json(left),base_gradients=matrix_json(grad),direction_gradients=matrix_json(g1),acceleration_gradients=matrix_json(g2),cubic_numerator=matrix_json(third),cubic_coefficient=ball_json(cubic),unfolding=matrix_json(unfold),unfolding_determinant=ball_json(determinant),parameter_solutions=parameter_solutions,deflated=matrix_json(deflated),spectral_exclusion=spectral,divisors=divisors,
        statement={'primitive_period':period,'dimension':n,'augmented_dimension':dim,'parameters':'actual hidden/output learning rates','scope':'local generic cusp of primitive periodic solutions; noncritical spectrum off unit circle; normalized augmented root locally unique','conventions':'q.q=1; q.h=0; w.q=1; h is second path derivative; cubic coefficient w(Cqqq+3Bqh)/6','not_established':['global cusp uniqueness','attracting periodic window','connection to prior fold or interior root','complete shrimp geometry']})
    observe('cusp_gates',{'passed':result['passed'],'gates':result['gates'],'cubic':result['cubic_coefficient'],'unfolding_determinant':result['unfolding_determinant']})
    return result
