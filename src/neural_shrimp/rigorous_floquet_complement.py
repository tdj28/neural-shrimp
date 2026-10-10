"""Fixed Floquet-scaled phase complements with rigorous uniform error bounds.

Floating eigensystems propose exact dyadic complement columns only. All native
root/critical-plane identities and whole-domain acquisition bindings belong to
the caller. Validated full frames then represent all remaining multipliers.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from flint import arb,arb_mat
from .rigorous_connection import require
from .rigorous_orbits import eye,midpoint_matrix,matrix_json,ball_json
from .rigorous_periodic_lyapunov import midpoint_float,exact_float_matrix
from .rigorous_norm_cocycle import operator_upper,frobenius_upper,advance,scalar_remainder_metric


def proposed_complements(blocks):
    p=len(blocks);require(1<=p<=18,'bounded phase count');n=blocks[0].nrows();require(4<=n<=13,'bounded native dimension')
    matrices=[midpoint_float(A)for A in blocks];M=np.eye(n)
    for A in matrices:
        require(A.shape==(n,n)and np.isfinite(A).all(),'finite full midpoint derivative');M=A@M
    values,vectors=np.linalg.eig(M);selected=sorted(range(n),key=lambda i:(abs(values[i]),values[i].real,values[i].imag))[:n-2]
    remaining=set(selected);columns=[];roots=[];groups=[]
    while remaining:
        i=min(remaining,key=lambda k:(values[k].real,abs(values[k].imag),values[k].imag));value=values[i];v=vectors[:,i]
        if abs(value.imag)<=1e-10*max(1.,abs(value)):
            require(np.linalg.norm(v.imag)<1e-7,'real proposed mode');u=v.real;u/=np.linalg.norm(u)
            columns.append(u);rho=max(abs(value.real),1e-12)**(1/p);roots.append(np.array([[rho]]));groups.append({'indices':[i],'type':'real','value':[float(value.real),float(value.imag)],'positive_root':rho});remaining.remove(i)
        else:
            partners=[j for j in remaining if j!=i];require(partners,'complete proposed conjugate pair')
            j=min(partners,key=lambda k:abs(values[k]-value.conjugate()));require(abs(values[j]-value.conjugate())<1e-7*max(1.,abs(value)),'matched conjugate proposal')
            if value.imag<0:i,j=j,i;value=values[i];v=vectors[:,i]
            scale=np.linalg.norm(v);u=v.real/scale;w=v.imag/scale;columns.extend([u,w]);rho=max(abs(value),1e-12)**(1/p);angle=np.angle(value)/p
            c,s=rho*np.cos(angle),rho*np.sin(angle);roots.append(np.array([[c,s],[-s,c]]));groups.append({'indices':[i,j],'type':'complex','value':[float(value.real),float(value.imag)],'root_radius':float(rho),'root_angle':float(angle)});remaining.remove(i);remaining.remove(j)
    require(len(columns)==n-2,'all proposed quotient dimensions');E=np.column_stack(columns);D=np.zeros((n-2,n-2));offset=0
    for block in roots:k=len(block);D[offset:offset+k,offset:offset+k]=block;offset+=k
    inv=np.linalg.inv(D);frames=[]
    for j in range(p):
        require(np.isfinite(E).all(),'finite propagated complement proposal');frames.append(exact_float_matrix(E))
        if j<p-1:E=matrices[j]@E@inv
    return frames,{'eigenvalues':[[float(v.real),float(v.imag)]for v in values],'selected_indices':selected,'mode_groups':groups,'root_matrix':D.tolist(),'phase_complements':[matrix_json(E)for E in frames],'cyclic_final_index':0,'scope':'Floating proposals only; smallest-modulus selection and root scalings supply no spectral proof. Negative real signs are carried by the final cyclic phase.'}


def balanced_complements(blocks,flat):
    """Orthogonal quotient and periodic Lyapunov metric propose exact columns.

    For midpoint quotient Q_j, P_j=I+Q_j^T P_(j+1) Q_j is solved cyclically.
    Cholesky B_j^T B_j=P_j yields E'_j=E_j B_j^-1. These are proposals only:
    final variable frames and every operator/remainder bound are revalidated.
    """
    p=len(blocks);n=blocks[0].nrows();m=n-2;matrices=[midpoint_float(A)for A in blocks]
    require(1<=p<=18 and 4<=n<=13 and len(flat)==3*n*p+2,'bounded full balanced-frame input')
    E=[];F=[];conditions=[]
    for j in range(p):
        off=3*n*j;critical=np.array([[float(flat[off+n+i].mid()),float(flat[off+2*n+i].mid())]for i in range(n)])
        left,singular,_=np.linalg.svd(critical,full_matrices=True)
        require(np.isfinite(critical).all()and singular[-1]>0,'full midpoint critical rank')
        comp=left[:,2:];frame=np.column_stack([critical,comp]);E.append(comp);F.append(frame)
        conditions.append({'phase':j,'critical_singular_values':singular.tolist(),'orthogonal_frame_condition':float(np.linalg.cond(frame))})
    Q=[(np.linalg.solve(F[(j+1)%p],matrices[j]@F[j]))[2:,2:]for j in range(p)]
    prefix=np.eye(m);forcing=np.zeros((m,m))
    for q in Q:forcing+=prefix.T@prefix;prefix=q@prefix
    p0=solve_discrete_lyapunov(prefix.T,forcing,method='direct');P=[None]*p;P[0]=(p0+p0.T)/2
    after=P[0]
    for j in range(p-1,0,-1):value=np.eye(m)+Q[j].T@after@Q[j];P[j]=(value+value.T)/2;after=P[j]
    out=[];factors=[]
    for j in range(p):
        require(np.isfinite(P[j]).all(),'finite cyclic Lyapunov proposal');B=np.linalg.cholesky(P[j]).T;basis=E[j]@np.linalg.inv(B)
        require(np.isfinite(basis).all(),'finite balanced complement proposal');out.append(exact_float_matrix(basis));factors.append(B)
        conditions[j]['balanced_frame_condition']=float(np.linalg.cond(np.column_stack([F[j][:,:2],basis])))
        conditions[j]['metric_condition']=float(np.linalg.cond(P[j]))
    return out,{'strategy':'periodic_lyapunov_balanced_orthogonal','conditions':conditions,'midpoint_quotients':[q.tolist()for q in Q],
      'midpoint_quotient_product':prefix.tolist(),'periodic_forcing':forcing.tolist(),'periodic_metrics':[v.tolist()for v in P],
      'upper_cholesky_factors':[v.tolist()for v in factors],'phase_complements':[matrix_json(e)for e in out],'cyclic_final_index':0,
      'scope':'All SVD/linear solves/periodic metrics are floating proposals only. Full native variable frames and every complementary mode are rigorously bounded afterward; no midpoint stability inference.'}


def uniform_complement(blocks,flat,p,n,terminal,*,tick=lambda:None,observe=lambda k,v:None):
    require(1<=p<=18 and 4<=n<=13 and len(blocks)==p and len(flat)==3*n*p+2,'full native phase/coordinate inventory')
    require(terminal.nrows()==terminal.ncols()==2,'critical terminal block')
    require(all(A.nrows()==A.ncols()==n and all(v.is_finite()for row in A.tolist()for v in row)for A in blocks),'finite native whole-domain derivatives')
    require(all(v.is_finite()for v in flat)and all(v>0 for v in flat[-2:]),'finite native box and actual rates')
    E,proposal=balanced_complements(blocks,flat);observe('proposal',proposal);frames=[];inverses=[]
    for j in range(p):
        tick();off=3*n*j;q=flat[off+n:off+2*n];r=flat[off+2*n:off+3*n]
        frame=arb_mat([[q[i],r[i]]+[E[j][i,k]for k in range(n-2)]for i in range(n)]);inverse=frame.inv()
        require(all(v.is_finite()for row in inverse.tolist()for v in row),'validated whole-domain critical/complement frame')
        frames.append(frame);inverses.append(inverse);observe('frame',{'phase':j,'full_frame':matrix_json(frame),'validated_inverse':matrix_json(inverse)})
    m=n-2;acc=([eye(m),arb_mat(m,m),arb_mat(m,m)],arb(0),arb(1));rows=[]
    for j,A in enumerate(blocks):
        tick();T=inverses[(j+1)%p]*A*frames[j];expected=terminal if j==p-1 else eye(2)
        require(all(T[i,k].contains(0)for i in range(2,n)for k in range(2)),'conditional native invariant two-plane enclosed')
        require(all(T[i,k].overlaps(expected[i,k])for i in range(2)for k in range(2)),'same conditional critical block enclosed')
        Q=arb_mat([[T[i,k]for k in range(2,n)]for i in range(2,n)]);center=midpoint_matrix(Q);error=Q-center
        op,gamma=operator_upper(center,tick);require(op['passed'],'validated midpoint operator bound');epsilon=frobenius_upper(error)
        phase=([center,arb_mat(m,m),arb_mat(m,m)],epsilon,gamma);acc,detail=advance(acc,phase)
        row={'phase':j,'full_transformed':matrix_json(T),'quotient':matrix_json(Q),'center':matrix_json(center),'matrix_error':matrix_json(error),'error_norm_upper':ball_json(epsilon),'operator_bound':op,'product_bound':detail};observe('phase',row);rows.append(row)
    metric=scalar_remainder_metric(*acc,tick=tick);observe('metric',metric)
    return {'passed':bool(metric['passed']),'dimension':n,'period':p,'complement_dimension':m,'exact_phase_complement_proposal':proposal,'quotient_phases':rows,'center_product':matrix_json(acc[0][0]),'product_error_norm_upper':ball_json(acc[1]),'center_product_norm_upper':ball_json(acc[2]),'metric':metric,'scope':'Conditional uniform attraction of all n−2 complementary native multipliers on the same independently proved root tube. Variable critical columns have validated full-domain inverses and exact cyclic phase closure; all operator errors retained. No signed NS criticality, stability side or completed window.'}
