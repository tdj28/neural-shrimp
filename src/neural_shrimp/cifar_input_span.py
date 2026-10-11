"""Saved-vector input-span algebra only: no model, derivative or trajectory evaluation."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)

def finite(a,shape=None):
    a=np.asarray(a)
    require(a.dtype==np.float64 and not np.iscomplexobj(a) and np.all(np.isfinite(a)),'finite float64 array')
    require(shape is None or a.shape==shape,'array shape')
    return a

def norm(a):return float(np.linalg.norm(np.asarray(a).ravel()))

def error_check(actual,predicted,scale=None):
    a=finite(actual);b=finite(predicted,a.shape)
    reference=norm(a) if scale is None else float(scale)
    require(np.isfinite(reference) and reference>=0,'finite nonnegative error scale')
    error=norm(a-b);limit=1e-12+1e-8*reference
    return dict(error=error,limit=limit,scale=reference,passed=bool(np.isfinite(error) and error<=limit))

@dataclass(frozen=True)
class Packing:
    d:int=3072
    width:int=32
    k:int=10
    @property
    def first(self):return self.d*self.width
    @property
    def size(self):return self.first+self.width+self.width*self.k+self.k
    def weight(self,v):return finite(v,(self.size,))[:self.first].reshape(self.d,self.width,order='C')
    def dimensions(self,columns):return dict(full=self.size,active=columns*self.width+self.size-self.first,complement=(self.d-columns)*self.width)

def decay(eta,lam,period=2):
    require(type(period) is int and period>=1,'positive integer period')
    require(all(isinstance(x,(float,int)) and np.isfinite(x) for x in (eta,lam)),'finite rates')
    eta,lam=float(eta),float(lam);exact_q=1-Fraction(eta)*Fraction(lam)
    q=float(1.-eta*lam);factor=float(q**period-1.)
    return dict(eta=eta,lam=lam,eta_hex=eta.hex(),lam_hex=lam.hex(),eta_exact=str(Fraction(eta)),lam_exact=str(Fraction(lam)),
                exact_q=str(exact_q),exact_factor=str(exact_q**period-1),rounded_q=q,rounded_q_hex=q.hex(),rounded_factor=factor,
                rounded_factor_hex=factor.hex(),period=period,elimination_valid=bool(exact_q**period!=1),omitted_directions_attracting=bool(abs(exact_q**period)<1))

def thin_basis(x):
    x=finite(x);require(x.ndim==2 and 0<x.shape[0]<=x.shape[1],'thin QR shape')
    q,r=np.linalg.qr(x.T,mode='reduced')
    require(q.shape==(x.shape[1],x.shape[0]) and r.shape==(x.shape[0],x.shape[0]),'retain all thin QR columns')
    return finite(q),finite(r)

def basis_checks(x,q,r):
    x=finite(x);q=finite(q,(x.shape[1],x.shape[0]));r=finite(r,(x.shape[0],x.shape[0]))
    gram=q.T@q;orth=norm(gram-np.eye(q.shape[1]));representation=norm(x-(x@q)@q.T)/max(1.,norm(x))
    # P(P(Q))-P(Q), evaluated through skinny factors, never a d-by-d projector.
    pq=q@gram;idempotence=norm(q@(q.T@pq)-pq)/max(1.,norm(pq))
    qr=norm(x.T-q@r)/max(1.,norm(x))
    values=dict(orthogonality_frobenius=orth,x_i_minus_p_relative=representation,representation_relative=representation,
                implicit_idempotence_on_q=idempotence,qr_reconstruction_relative=qr,retained_columns=q.shape[1],rank_truncation=False)
    values['passed']=bool(orth<=1e-10 and representation<=1e-12 and idempotence<=1e-10 and qr<=1e-12)
    return values

def project(v,q,packing):
    w=packing.weight(v);q=finite(q);require(q.ndim==2 and q.shape[0]==packing.d,'basis dimension')
    active=v.copy();active[:packing.first]=(q@(q.T@w)).ravel(order='C')
    perpendicular=np.zeros_like(v);perpendicular[:packing.first]=v[:packing.first]-active[:packing.first]
    return finite(active),finite(perpendicular)

def vector_checks(v,active,perpendicular,x,q,packing):
    v=finite(v,(packing.size,));active=finite(active,v.shape);perpendicular=finite(perpendicular,v.shape)
    n=norm(v);pp,_=project(active,q,packing)
    w=packing.weight(v);n_w=packing.weight(perpendicular)
    checks={'reconstruction':error_check(active+perpendicular,v,n),
            'implicit_projector_idempotence':error_check(pp,active,n),
            'complement_orthogonality':error_check(q.T@n_w,np.zeros((q.shape[1],packing.width)),n),
            'x_complement_preactivation':error_check(x@n_w,np.zeros((x.shape[0],packing.width)),norm(x@w))}
    rem=packing.first
    require(np.array_equal(active[rem:],v[rem:]) and np.array_equal(perpendicular[rem:],np.zeros_like(v[rem:])),'non-first-layer components remain active')
    active_n,perp_n=norm(active),norm(perpendicular)
    squared_error=abs(n*n-active_n*active_n-perp_n*perp_n);squared_limit=1e-12+1e-8*n*n
    return dict(norm=n,active_norm=active_n,complement_norm=perp_n,
      complementary_squared_fraction=perp_n*perp_n/(n*n) if n>0 else None,
      active_squared_fraction=active_n*active_n/(n*n) if n>0 else None,
      pythagorean_error=squared_error,pythagorean_limit=squared_limit,checks=checks,
      passed=bool(squared_error<=squared_limit and all(c['passed'] for c in checks.values())))

def identities(residual,perpendicular_residual,state_perpendicular,factor):
    expected=factor*state_perpendicular
    return expected,error_check(perpendicular_residual,expected,norm(residual))

def linear_identities(vectors,parts,factor):
    # Only at the original basepoint; no trial-state Jacobian action is available.
    p,ap,e,r=(vectors[n] for n in ('p','ap','e','original-residual'))
    n=parts['original-state'][1];pp,ap_perp,ep,rp=(parts[s][1] for s in ('p','ap','e','original-residual'))
    active={s:parts[s][0] for s in ('ap','e','original-residual')}
    ap_prediction=factor*pp;e_prediction=factor*(pp+n)
    checks={'ap_complement_identity':error_check(ap_perp,ap_prediction,norm(ap)),
            'e_complement_identity':error_check(ep,e_prediction,norm(e)),
            'e_equals_ap_plus_r':error_check(e,ap+r,norm(e)),
            'e_active_equals_ap_active_plus_r_active':error_check(active['e'],active['ap']+active['original-residual'],norm(e)),
            'e_complement_equals_ap_complement_plus_r_complement':error_check(ep,ap_perp+rp,norm(e))}
    checks['saved_sum_bitwise_equal']=bool(e.tobytes()==(ap+r).tobytes())
    return ap_prediction,e_prediction,dict(checks=checks,passed=bool(checks['saved_sum_bitwise_equal'] and all(v['passed'] for v in checks.values() if isinstance(v,dict))))


def saved_relations(theta,vectors,map2,trials):
    """Validate saved associations without evaluating the training map."""
    labels=('original','trial-0','trial-1','trial-2','trial-3')
    expected=[label+'-'+what for label in labels for what in ('state','residual')]+['p','ap','e']
    require(list(vectors)==expected and list(map2)==list(labels),'declared state/vector order')
    theta=finite(theta);require(theta.ndim==1,'flat state')
    for label in labels:
        state=finite(vectors[label+'-state'],theta.shape);r=finite(vectors[label+'-residual'],theta.shape)
        require(r.tobytes()==(finite(map2[label],theta.shape)-state).tobytes(),'saved residual ordering '+label)
    for name in ('p','ap','e'):finite(vectors[name],theta.shape)
    require(theta.tobytes()==vectors['original-state'].tobytes(),'saved original theta')
    require(len(trials)==4,'exactly four trial records')
    for i,alpha in enumerate((1.,.5,.25,.125)):
        require(trials[i]['index']==i and trials[i]['alpha']==alpha,'saved trial order')
        require(vectors[f'trial-{i}-state'].tobytes()==(theta+alpha*vectors['p']).tobytes(),'saved trial/base/direction relation')
    require(vectors['e'].tobytes()==(vectors['ap']+vectors['original-residual']).tobytes(),'saved e=Ap+R identity')
