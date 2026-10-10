"""Conditional transverse cocycle from a proved native periodic +1 mode.

At the exact authenticated roots J_j q_j=q_(j+1), q_p=q_0. For any invertible
V_j=[q_j,C_j], T_j=V_(j+1)^-1 J_j V_j has first column e1 THERE. Its lower
right block describes the quotient by the neutral direction. No such identity
is asserted for independent arbitrary interval entries.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,norm_inf_upper,ball_json,matrix_json,vector_json
from .rigorous_shooting_adjoint import need,finite
from .rigorous_periodic_lyapunov import periodic_witness


def coordinate_frame(q,observe=lambda kind,value:None):
    n=len(q);need(2<=n<=13 and all(v.is_finite() for v in q),'finite native mode')
    # Fixed proposal rule: largest exact midpoint component, first index on tie.
    pivot=max(range(n),key=lambda i:abs(q[i].mid()))
    need(not q[pivot].contains(0),'selected mode coordinate excludes zero')
    others=[i for i in range(n) if i!=pivot]
    v=arb_mat(n,n);inverse=arb_mat(n,n)
    for i in range(n):v[i,0]=q[i]
    inverse[0,pivot]=1/q[pivot]
    for k,i in enumerate(others,1):
        v[i,k]=1;inverse[k,i]=1;inverse[k,pivot]=-q[i]/q[pivot]
    observe('frame_return',{'q':vector_json(q),'pivot':pivot,'complement_indices':others,'V':matrix_json(v),'inverse':matrix_json(inverse)})
    need(finite(v) and finite(inverse),'finite frame return')
    defect=norm_inf_upper(eye(n)-inverse*v)
    observe('frame_defect_return',{'norm_inf_upper':ball_json(defect)})
    need(defect.is_finite() and defect<1,'frame inverse residual gate')
    return v,inverse,{'pivot':pivot,'complement_indices':others,'inverse_defect':ball_json(defect)}


def quotient_blocks(jacobians,modes,observe=lambda kind,value:None):
    p=len(jacobians);need(1<=p<=18 and len(modes)==p,'fixed cyclic phases')
    n=len(modes[0]);need(2<=n<=13 and all(len(q)==n for q in modes),'mode dimensions')
    need(all(j.nrows()==j.ncols()==n and finite(j)for j in jacobians),'finite native Jacobians')
    frames=[];inverses=[];meta=[]
    for j,q in enumerate(modes):
        observe('frame_start',{'phase':j})
        v,vi,m=coordinate_frame(q,observe);frames.append(v);inverses.append(vi);meta.append(m)
    blocks=[]
    for j,a in enumerate(jacobians):
        transformed=inverses[(j+1)%p]*a*frames[j]
        observe('transformed_return',{'phase':j,'next_phase':(j+1)%p,'J':matrix_json(a),'T':matrix_json(transformed)})
        need(finite(transformed),'finite transformed matrix')
        # This consistency check does not prove the exact-root mode premise.
        need(all(transformed[i,0].contains(int(i==0))for i in range(n)),'exact-root first-column compatibility')
        b=arb_mat([[transformed[i,jj]for jj in range(1,n)]for i in range(1,n)])
        observe('quotient_return',{'phase':j,'matrix':matrix_json(b)});blocks.append(b)
    return blocks,meta


def qualify(jacobians,modes,*,precision=256,tick=lambda:None,observe=lambda kind,value:None):
    blocks,frames=quotient_blocks(jacobians,modes,observe)
    metric=periodic_witness(blocks,precision=precision,tick=tick,observe=observe)
    observe('periodic_metric_return',metric)
    return {'schema':'neural-shrimp-native-neutral-quotient-v1','passed':metric['passed'],'period':len(blocks),'native_dimension':len(modes[0]),'quotient_dimension':blocks[0].nrows(),'frames':frames,'metric':metric,
      'exact_root_premise':'Authenticated native periodic normalized +1 eigenpair; J_j q_j=q_(j+1), q_p=q_0. First column e1 is conditional at those roots, not box-wide.',
      'theorem':'Cyclic invertible V_j=[q_j,C_j] make the exact native monodromy similar to an upper triangular matrix with 1 and the quotient product. Strict cyclic quotient metrics place every other multiplier inside the unit circle.',
      'scope':'Attracting complement only; the exact neutral+1 prevents attraction of the fold cycle. Root existence/primitive identity, simple mode and nonlinear coefficients are separately inherited premises.'}
