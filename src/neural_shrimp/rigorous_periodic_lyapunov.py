"""Periodic quadratic witnesses from saved interval phase Jacobians.

Floating linear algebra only proposes exact dyadic metrics/congruences. Strict
Arb Gershgorin bounds after invertible congruence decide positive definiteness.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,dyadic,eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json
from .rigorous_connection import require


def exact_float_matrix(array):
    require(np.isfinite(array).all(),'finite floating proposal')
    return arb_mat([[exact(dyadic(float(x))) for x in row] for row in array])


def midpoint_float(matrix):
    return np.array([[float(matrix[i,j].mid()) for j in range(matrix.ncols())]
                     for i in range(matrix.nrows())],dtype=np.float64)


def propose_metrics(blocks):
    points=[midpoint_float(m) for m in blocks];n=points[0].shape[0]
    prefix=np.eye(n);forcing=np.zeros((n,n))
    for m in points:
        forcing+=prefix.T@prefix;prefix=m@prefix
    # scipy solves A X A^T-X+Q=0; A=M^T gives P-M^T P M=Q.
    first=solve_discrete_lyapunov(prefix.T,forcing,method='direct')
    metrics=[None]*len(points);metrics[0]=(first+first.T)/2
    next_metric=metrics[0]
    for j in range(len(points)-1,0,-1):
        value=points[j].T@next_metric@points[j]+np.eye(n)
        metrics[j]=(value+value.T)/2;next_metric=metrics[j]
    return [exact_float_matrix(p) for p in metrics]


def positive_definite(matrix,tick=lambda:None):
    """Every real symmetric member is PD if all enclosed row margins are >0."""
    n=matrix.nrows();require(matrix.ncols()==n,'square symmetric enclosure')
    mid=midpoint_float(matrix);mid=(mid+mid.T)/2
    out={'passed':False}
    try:
        tick();basis=exact_float_matrix(np.linalg.cholesky(mid))
        inverse=basis.inv();pre=midpoint_matrix(inverse)
        defect=norm_inf_upper(eye(n)-pre*basis)
        require(defect.is_finite() and defect<1,'congruence inverse certified')
        changed=inverse*matrix*inverse.transpose()
        margins=[changed[i,i].lower()-sum((abs(changed[i,j]).upper() for j in range(n) if j!=i),arb(0)) for i in range(n)]
        out.update(passed=all(v.is_finite() and v>0 for v in margins),basis=matrix_json(basis),
            inverse=matrix_json(inverse),inverse_defect=ball_json(defect),congruence=matrix_json(changed),
            row_margins=[ball_json(v) for v in margins])
    except (ValueError,ZeroDivisionError,np.linalg.LinAlgError) as exc:
        out['error']={'type':type(exc).__name__,'message':str(exc)}
    return out


def periodic_witness(blocks,*,precision=256,tick=lambda:None,observe=lambda k,r:None):
    require(1<=len(blocks)<=18 and all(m.nrows()==m.ncols()==blocks[0].nrows() for m in blocks),'fixed phase inventory')
    require(blocks[0].nrows()<=13 and 128<=precision<=512,'dimension/precision cap')
    with ctx.workprec(precision):
        tick();metrics=propose_metrics(blocks);n=blocks[0].nrows();rows=[]
        point_pass=True
        for j,(block,p) in enumerate(zip(blocks,metrics)):
            pp=metrics[(j+1)%len(metrics)];mid=midpoint_matrix(block)
            d0=p-mid.transpose()*pp*mid
            pcheck=positive_definite(p,tick);dcheck=positive_definite(d0,tick)
            tick();condition=norm_inf_upper(p)*norm_inf_upper(p.inv())
            defect=norm_inf_upper(d0-eye(n))
            guard=condition.is_finite() and condition<2**40 and defect.is_finite() and defect<arb('1/100')
            passed=bool(pcheck['passed'] and dcheck['passed'] and guard)
            row={'phase':j,'P':matrix_json(p),'point_J':matrix_json(mid),'point_D':matrix_json(d0),
                 'P_positive':pcheck,'point_D_positive':dcheck,'condition_inf_upper':ball_json(condition),
                 'point_defect_from_identity':ball_json(defect),'proposal_guard':bool(guard),'point_passed':passed}
            rows.append(row);point_pass=point_pass and passed
        observe('point_metrics',{'point_passed':point_pass,'metrics':rows})
        if point_pass:
            for j,(block,row) in enumerate(zip(blocks,rows)):
                p=metrics[j];pp=metrics[(j+1)%len(metrics)];mid=midpoint_matrix(block);error=block-mid
                # Exact expansion retains small perturbations around the midpoint derivative.
                d0=p-mid.transpose()*pp*mid
                decrease=d0-mid.transpose()*pp*error-error.transpose()*pp*mid-error.transpose()*pp*error
                check=positive_definite(decrease,tick)
                row['interval_D']=matrix_json(decrease);row['interval_D_positive']=check
        passed=bool(point_pass and all(row['interval_D_positive']['passed'] for row in rows))
        return {'schema':'neural-shrimp-periodic-quadratic-witness-v1','passed':passed,'point_passed':bool(point_pass),
            'precision':precision,'phase_count':len(blocks),'dimension':n,'cyclic_P_final_index':0,
            'interval_J':[matrix_json(m) for m in blocks],'phases':rows,
            'theorem':'Each fixed P_j is positive definite and P_j-J_j^T P_(j+1) J_j is positive definite for every enclosed native J_j. Cyclic P_p=P_0 gives strict contraction over one full period in the P_0 quadratic norm.',
            'scope':'stability witness for the supplied phase-Jacobian enclosures only; orbit existence, primitive period and branch joins are separate prerequisites',
            'trust':'Arb enclosures plus this analytic identity; numerical Lyapunov/Cholesky only propose exact dyadic matrices'}
