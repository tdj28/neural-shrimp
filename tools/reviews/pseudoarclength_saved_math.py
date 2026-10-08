"""Independent arithmetic on retained pseudo-arclength arrays only.

No model, derivative, decomposition, assignment solver, or linear solve calls.
"""
import importlib.util
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('independent_lower_gap_audit',ROOT/'tools/reviews/audit_s1r_lower_gap_outcome.py')
G=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(G)
require,array,exact,scalar,compare,close_array=G.require,G.array,G.exact,G.scalar,G.compare,G.close_array
norm,dot=G.norm,G.dot


def complex_array(value,shape=None):
    value=np.asarray(value)
    require(value.dtype in (np.float64,np.complex128) and (shape is None or value.shape==shape) and bool(np.isfinite(value).all()),'finite real/complex saved array')
    return value

def cnorm(value):
    value=complex_array(value)
    result=math.sqrt(math.fsum(float(x.real)**2+float(x.imag)**2 for x in value.flat))
    require(math.isfinite(result),'finite complex norm');return result

def cdot(left,right):
    left,right=complex_array(left),complex_array(right);require(left.shape==right.shape,'complex dot shape')
    values=[complex(x).conjugate()*complex(y) for x,y in zip(left.flat,right.flat)]
    return complex(math.fsum(z.real for z in values),math.fsum(z.imag for z in values))

def scale(*states):return max(1.,*(float(np.max(np.abs(array(x)))) for x in states))
def distance(a,b):return float(np.max(np.abs(array(a)-array(b))))/scale(a,b)
def normalized(a,b):return float(np.max(np.abs(array(a)-array(b))/(1e-9+1e-8*np.maximum(np.abs(a),np.abs(b)))))

def phase_match(parent,child,replay=False):
    parent,child=array(parent),array(child);require(parent.shape==child.shape and parent.ndim==2 and len(parent)==2,'full native pairs')
    costs=[distance(parent,child),distance(parent,child[::-1])];phase=int(costs[1]<costs[0]);separation=float(np.max(np.abs(parent[0]-parent[1])))/scale(parent)
    cap=min(.01,.25*separation,1e-7 if replay else math.inf);gap=costs[1-phase]-costs[phase]
    return dict(passed=bool(costs[phase]<=cap and gap>=.5*separation and separation>=1e-6),costs=costs,phase=phase,separation=separation,displacement_cap=cap,cost_gap=gap,replay=bool(replay))

def state_checks(pair,images,second):
    pair=array(pair);images=array(images,pair.shape);second=array(second,pair.shape)
    require(pair.ndim==2 and len(pair)==2,'two native phases')
    closure=[normalized(second[i],pair[i]) for i in range(2)];transition=[normalized(images[i],pair[1-i]) for i in range(2)]
    separation=float(np.max(np.abs(pair[0]-pair[1])))/scale(pair)
    return dict(passed=bool(max(closure+transition)<=1 and separation>=1e-6),closure=closure,transitions=transition,separation=separation)

def phase_arithmetic(theta,h,output,returned,hidden=64):
    theta=array(theta);n=len(theta);rates=np.r_[np.full(hidden,10.**float(h)),np.full(n-hidden,10.**float(output))]
    require(bool(np.isfinite(rates).all()) and bool(np.all(rates>0)),'positive finite actual rates')
    return G.phase_check(theta,rates,returned,returned,hidden=hidden)

def assembly(theta,first,second):
    theta=array(theta);n=len(theta)
    require(exact(first['theta'],theta) and exact(second['theta'],first['image']),'actual two-step native inputs')
    J0=array(first['jacobian'],(n,n));J1=array(second['jacobian'],(n,n));c0=array(first['parameter'],(n,));c1=array(second['parameter'],(n,))
    M=J1@J0
    return dict(R=array(second['image'],(n,))-theta,M=M,A=M-np.eye(n),b=J1@c0+c1,J0=J0,partial_h_F=c0)

def bordered(z,R,A,b,parent,tangent,target):
    z=array(z);n=len(z)-1;parent=array(parent,z.shape);tangent=array(tangent,z.shape)
    plane=dot(tangent,z-parent)-float(target)
    return np.r_[array(R,(n,)),plane],np.block([[array(A,(n,n)),array(b,(n,))[:,None]],[tangent[None,:]]])

def linear_check(Gv,DG,step,z):
    Gv=array(Gv);n=len(Gv);DG=array(DG,(n,n));step=array(step,(n,));z=array(z,(n,))
    error=DG@step+Gv;absolute=norm(error);gnorm=norm(Gv);near=1e-12*max(1.,norm(z));denom=norm(DG)*norm(step)+gnorm
    backward=absolute/denom if denom>0 else (0. if absolute==0 else None)
    forcing=absolute/gnorm if gnorm>near else None
    passed=backward is not None and backward<=1e-12 and ((forcing is not None and forcing<=1e-6) or (forcing is None and absolute<=1e-12))
    return dict(error=error,absolute_linear_residual=absolute,true_relative_forcing=forcing,nearzero_threshold=near,residual_norm=gnorm,normwise_backward_error_frobenius=backward,passed=bool(passed),slope=dot(Gv,DG@step),merit=.5*dot(Gv,Gv))

def trial_decision(Gv,DG,step,trial_G,halving):
    require(type(halving) is int and 0<=halving<=12,'fixed halving range')
    alpha=2.**-halving;slope=dot(array(Gv),array(DG)@array(step));merit=.5*dot(Gv,Gv);trial_merit=.5*dot(trial_G,trial_G);bound=merit+1e-4*alpha*slope
    return dict(alpha=alpha,slope=slope,merit=merit,trial_merit=trial_merit,armijo_bound=bound,accepted=bool(slope<0 and trial_merit<=bound))

def locality(z,predictor,seed,anchor):
    z,predictor,seed,anchor=(array(x) for x in (z,predictor,seed,anchor));require(z.shape==predictor.shape==seed.shape==anchor.shape,'augmented locality shapes')
    ell=norm(predictor-seed);correction=norm(z-predictor);theta_distance=distance(z[:-1],predictor[:-1]);anchor_theta=norm(z[:-1]-anchor[:-1]);anchor_h=abs(float(z[-1]-anchor[-1]))
    return dict(ell=ell,correction_distance=correction,candidate_distance=theta_distance,anchor_theta_distance=anchor_theta,anchor_h_distance=anchor_h,passed=bool(ell>0 and correction<=.5*ell and theta_distance<=.02 and anchor_theta<=.02 and anchor_h<=.001))

def predictor(seed,tangent,parent,plane_tangent,target):
    seed,tangent,parent,plane_tangent=(array(x) for x in (seed,tangent,parent,plane_tangent));denominator=dot(plane_tangent,tangent)
    require(denominator>1e-8,'positive nonambiguous tangent-plane denominator')
    length=(float(target)-dot(plane_tangent,seed-parent))/denominator
    return seed+length*tangent,dict(denominator=denominator,length=length)

def tangent_check(B,saved,orientation):
    B=array(B);m,n=B.shape;require(n==m+1,'one-dimensional augmented kernel')
    diagnostics=svd_residuals(B,saved);rank=diagnostics['rank'];require(rank['status']=='full_row_rank_resolved','resolved full-row rank')
    v=array(saved['Vh'],(n,n))[-1].copy();v/=norm(v);orientation=array(orientation,(n,));length=norm(orientation)
    require(length>0,'nonzero tangent orientation');cosine=dot(v,orientation)/length
    require(abs(cosine)>1e-8,'nonambiguous tangent overlap')
    if cosine<0:v=-v;cosine=-cosine
    residual=norm(B@v)/max(1.,norm(B));# Numerical gate is assessed by the caller; failed retained gates remain auditable.
    return dict(tangent=v,secant_cosine=cosine,raw_residual=residual,h_component=float(v[-1]),rank=rank)

def svd_residuals(matrix,saved):
    matrix=array(matrix);m,n=matrix.shape;U=array(saved['U'],(m,m));s=array(saved['singular_values'],(min(m,n),));Vh=array(saved['Vh'],(n,n))
    require(m<=n,'declared square/augmented SVD');close_array(saved['matrix'],matrix,'SVD actual input')
    scale_=max(1.,norm(matrix));V=Vh[:m].T
    right_matrix=matrix@V-U*s;left_matrix=matrix.T@U-V*s
    right=np.array([norm(right_matrix[:,i])/scale_ for i in range(m)]);left=np.array([norm(left_matrix[:,i])/scale_ for i in range(m)])
    reconstruction=norm(matrix-(U*s)@Vh[:m])/scale_;orthogonal=max(norm(U.T@U-np.eye(m)),norm(Vh@Vh.T-np.eye(n)))
    # Return diagnostics even when the declared scientific residual gate fails.
    return dict(right_residual=right,left_residual=left,reconstruction=reconstruction,orthogonality=orthogonal,rank=G.rank(s))

def spectrum(matrix,saved,norm_svd,condition_svd):
    matrix=array(matrix);n=len(matrix);svd_residuals(matrix,norm_svd);svd_residuals(matrix-np.eye(n),condition_svd)
    values=complex_array(saved['values'],(n,));left=complex_array(saved['left'],(n,n));right=complex_array(saved['right'],(n,n))
    matrix_norm=float(norm_svd['singular_values'][0]);rr=[];lr=[];conditions=[]
    for i,mu in enumerate(values):
        rn,ln=cnorm(right[:,i]),cnorm(left[:,i]);require(rn>0 and ln>0,'nonzero left/right vectors');denom=max(np.finfo(float).tiny,matrix_norm+abs(complex(mu)))
        rr.append(cnorm(matrix@right[:,i]-mu*right[:,i])/(denom*rn));lr.append(cnorm(matrix.T@left[:,i]-np.conj(mu)*left[:,i])/(denom*ln));overlap=abs(cdot(left[:,i],right[:,i]))/(ln*rn)
        conditions.append(1/overlap if overlap>0 and math.isfinite(1/overlap) else None)
    # Retained failing eigenpairs are evidence of an unresolved qualifier.
    s=condition_svd['singular_values'];condition=float(s[0]/s[-1]) if s[-1]>0 else None;radius=max(abs(complex(mu)) for mu in values)
    return dict(norm=matrix_norm,right_residual=np.array(rr),left_residual=np.array(lr),eigenvalue_condition=conditions,condition_m_minus_i=condition,min_spectral_gap=min(abs(complex(values[i]-values[j])) for i in range(n) for j in range(i)) if n>1 else None,radius=radius,stability='attracting' if radius<1-1e-7 else 'unstable' if radius>1+1e-7 else 'neutral_or_unresolved')

def spectral_assignment(first,second,assignment):
    first,second=complex_array(first),complex_array(second);n=len(first)
    require(second.shape==first.shape and len(assignment)==n and sorted(int(i) for i in assignment)==list(range(n)) and all(type(i) is int or isinstance(i,np.integer) for i in assignment),'complete spectral bijection')
    error=max(abs(complex(first[i]-second[int(j)])) for i,j in enumerate(assignment));require(error<=1e-7,'native full-spectrum matching')
    return error

def transport(first_tangent,second_tangent,J0,c0,B1):
    first_tangent,second_tangent=array(first_tangent),array(second_tangent);v=np.r_[array(J0)@first_tangent[:-1]+array(c0)*first_tangent[-1],first_tangent[-1]];length=norm(v);require(length>0,'nonzero transported tangent')
    v/=length;residual=norm(array(B1)@v)/max(1.,norm(B1));cosine=dot(v,second_tangent)
    # Return the actual transport mismatch; acceptance is checked separately.
    return dict(normalized=v,residual=residual,cosine=cosine)
