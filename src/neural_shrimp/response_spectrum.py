"""Finite-window response diagnostics, not a cavity closure or certified spectrum.

K v at time t is v[t+1]-exp(-s) M[t]v[t]. The free window leaves
both endpoints unknown; causal fixes v[0]=0; periodic identifies endpoints.
Time is optimizer steps. The infinite-time identity requires invertible maps,
a nonmarginal threshold and its own limit assumptions (Clark 2026, sec. 2).
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr, splu

BOUNDARIES = ("free", "periodic", "causal")


def checked(maps):
    a = np.asarray(maps, dtype=float)
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1] < 1 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        raise ValueError("nonempty finite square phase matrices required")
    return a


def layout(period, steps, boundary):
    if type(steps) is not int or steps < 2 or steps % period or boundary not in BOUNDARIES:
        raise ValueError("steps must be a positive whole number of periods >=2")
    return steps + (boundary == "free")


def indices(t, steps, boundary):
    if boundary == "causal":
        return t, t-1 if t else None
    return (t+1) % steps if boundary == "periodic" else t+1, t


def difference_matrix(maps, steps, shift, boundary):
    a = checked(maps); p, n, _ = a.shape; states = layout(p, steps, boundary)
    if not np.isfinite(shift) or abs(shift) > 100:
        raise ValueError("finite bounded shift required")
    scale = float(np.exp(-shift)); rows=[]; cols=[]; data=[]
    rr, cc = np.indices((n,n))
    for t in range(steps):
        nxt, prev = indices(t, steps, boundary)
        rows.extend(t*n+np.arange(n)); cols.extend(nxt*n+np.arange(n)); data.extend(np.ones(n))
        if prev is not None:
            rows.extend((t*n+rr).ravel()); cols.extend((prev*n+cc).ravel()); data.extend((-scale*a[t%p]).ravel())
    return sparse.coo_matrix((data,(rows,cols)), shape=(steps*n,states*n)).tocsc()


def difference_operator(actions, adjoints, dimension, steps, shift, boundary, tick=lambda:None):
    """Callback K/K^T: callbacks may later be JVP/VJP, without a stored Hessian."""
    p=len(actions); states=layout(p,steps,boundary)
    if len(adjoints)!=p or dimension<1 or not np.isfinite(shift) or abs(shift)>100:
        raise ValueError("invalid callback operator")
    n=dimension; scale=float(np.exp(-shift)); counts={"K":0,"KT":0,"JVP":0,"VJP":0}
    def mv(v):
        tick(); counts['K']+=1; v=np.asarray(v).reshape(states,n); y=np.empty((steps,n))
        for t in range(steps):
            nxt,prev=indices(t,steps,boundary); y[t]=v[nxt]
            if prev is not None:
                counts['JVP']+=1; y[t]-=scale*np.asarray(actions[t%p](v[prev]))
        return y.ravel()
    def rmv(w):
        tick(); counts['KT']+=1; w=np.asarray(w).reshape(steps,n); y=np.zeros((states,n))
        for t in range(steps):
            nxt,prev=indices(t,steps,boundary); y[nxt]+=w[t]
            if prev is not None:
                counts['VJP']+=1; y[prev]-=scale*np.asarray(adjoints[t%p](w[t]))
        return y.ravel()
    return LinearOperator((steps*n,states*n),matvec=mv,rmatvec=rmv,dtype=float), counts


def response(maps, steps, shift, eta, boundary, tick=lambda:None):
    """Exact-coordinate trace (all n RHS), sparse augmented solve, true residual."""
    a=checked(maps); p,n,_=a.shape
    if not np.isfinite(eta) or eta<=0: raise ValueError("positive finite eta required")
    start=time.monotonic(); tick(); k=difference_matrix(a,steps,shift,boundary); m,d=k.shape
    # eta*u + K*v = I; -K^T*u + eta*v=0 gives the regularized LS solution.
    b=sparse.bmat([[eta*sparse.eye(m),k],[-k.T,eta*sparse.eye(d)]],format='csc')
    phase=(steps//2//p)*p
    rhs=np.zeros((m+d,n)); rhs[phase*n:(phase+1)*n]=np.eye(n)
    lu=splu(b); sol=lu.solve(rhs); defect=b@sol-rhs
    v=sol[m:]; nxt,_=indices(phase,steps,boundary); block=v[nxt*n:(nxt+1)*n].copy()
    rel=float(np.linalg.norm(defect)/np.linalg.norm(rhs))
    result={"steps":steps,"periods":steps//p,"shift":float(shift),"eta":float(eta),"boundary":boundary,
            "impulse_time":phase,"readout_time":phase+1,"response_block":block,
            "count":float(np.trace(block)),"cdf":float(np.trace(block)/n),
            "true_augmented_relative_residual":rel,"solve_qualified":bool(np.isfinite(sol).all() and rel<=1e-8),
            "solution_norm":float(np.linalg.norm(sol)),"system_shape":b.shape,"system_nnz":b.nnz,
            "factor_nnz":lu.L.nnz+lu.U.nnz,"elapsed_seconds":time.monotonic()-start}
    # Caller persists this return before checking time/budget again.
    return result


def matrix_free_probe(maps, steps, shift, eta, boundary, maxiter=300, tick=lambda:None):
    a=checked(maps); p,n,_=a.shape
    if not np.isfinite(eta) or eta<=0: raise ValueError("positive finite eta")
    actions=[lambda v,m=m:m@v for m in a]; adjoints=[lambda v,m=m:m.T@v for m in a]
    k,counts=difference_operator(actions,adjoints,n,steps,shift,boundary,tick)
    phase=(steps//2//p)*p; rhs=np.zeros(steps*n); rhs[phase*n]=1
    start=time.monotonic(); ret=lsmr(k,rhs,damp=eta,atol=1e-10,btol=1e-10,conlim=1e12,maxiter=maxiter)
    v=ret[0]; residual=k@v-rhs; normal=k.T@residual+eta**2*v
    nxt,_=indices(phase,steps,boundary)
    return {"readout_column":v[nxt*n:(nxt+1)*n].copy(),"solution":v,"istop":ret[1],"iterations":ret[2],
            "reported_normr":ret[3],"reported_normar":ret[4],"true_normal_residual_norm":float(np.linalg.norm(normal)),
            "true_data_residual_norm":float(np.linalg.norm(residual)),"counts":counts,
            "elapsed_seconds":time.monotonic()-start,"maxiter":maxiter,
            "interpretation":"single e0 impulse; not a trace estimate or demonstrated HVP speedup"}


def references(maps, qr_periods=(128,512)):
    a=checked(maps); p,n,_=a.shape; mon=np.eye(n)
    for m in a: mon=m@mon
    vals,vec=np.linalg.eig(mon); modulus=np.abs(vals)
    eigres=np.linalg.norm(mon@vec-vec*vals,axis=0)/np.maximum(1,np.linalg.norm(mon,ord='fro'))
    exponents=np.full(n,np.nan); nonzero=modulus>0; exponents[nonzero]=np.log(modulus[nonzero])/p
    sv=[np.linalg.svd(m,compute_uv=False) for m in a]
    invertible=all(s[-1]>1e-12*max(1,s[0]) for s in sv)
    q=np.eye(n); sums=np.zeros(n); snapshots={}
    for j in range(1,max(qr_periods)+1):
        for m in a:
            q,r=np.linalg.qr(m@q); diag=np.abs(np.diag(r))
            if np.any(diag==0): sums[:]=np.nan
            else:sums+=np.log(diag)
        if j in qr_periods:snapshots[str(j)]=np.sort(sums/(j*p))
    return {"monodromy":mon,"multipliers_real":vals.real,"multipliers_imag":vals.imag,"multiplier_angles":np.angle(vals),
            "multipliers_modulus":modulus,"exponents_per_step":exponents,"eigen_residuals":eigres,
            "phase_smallest_singular_values":[s[-1] for s in sv],"numerically_invertible_phases":invertible,
            "qr_sorted_per_step":snapshots,"decompositions":{"eig":1,"phase_svd":p,"qr":max(qr_periods)*p},
            "reference_scope":"floating periodic cocycle; eig residuals do not certify ill-conditioned modes"}


def classify_count(ref, shift, count):
    ex=np.asarray(ref['exponents_per_step']); qr=np.asarray(ref['qr_sorted_per_step'][str(max(map(int,ref['qr_sorted_per_step'])))])
    finite=bool(np.isfinite(ex).all()); gap=float(np.min(abs(ex-shift))) if finite else None
    exact=int(np.sum(ex<shift)) if finite else None
    qr_count=int(np.sum(qr<shift)) if np.isfinite(qr).all() else None
    return {"floquet_count":exact,"qr_count":qr_count,"threshold_gap":gap,
            "identity_assumptions_numerically_screened":bool(ref['numerically_invertible_phases'] and finite and gap>1e-10),
            "count_error":None if exact is None else count-exact,
            "floquet_qr_count_agree":exact==qr_count,
            "finite_window_result_is_not_a_certificate":True}
