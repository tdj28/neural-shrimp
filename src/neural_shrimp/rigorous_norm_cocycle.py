"""Fixed cyclic frames and operator-norm cocycle remainders from saved jets.

All floating decompositions only propose exact matrices or norm bounds. Strict
Arb inequalities establish every inverse, operator bound and final decrease.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from flint import arb,arb_mat
from .rigorous_connection import require
from .rigorous_orbits import exact,dyadic,eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json
from .rigorous_periodic_lyapunov import midpoint_float,exact_float_matrix,positive_definite
from .rigorous_correlated_cocycle import phase_model,polynomial_value


def frobenius_upper(matrix):
    return sum((abs(v).upper()**2 for row in matrix.tolist() for v in row),arb(0)).sqrt().upper()


def operator_upper(matrix,tick=lambda:None):
    tick();proposal=float(np.linalg.svd(midpoint_float(matrix),compute_uv=False)[0])
    require(np.isfinite(proposal) and proposal>=0,'finite operator proposal')
    gamma=exact(dyadic(proposal*(1+2**-20)+2**-40))
    defect=eye(matrix.nrows())*(gamma*gamma)-matrix.transpose()*matrix
    check=positive_definite(defect,tick)
    return {'passed':check['passed'],'gamma':ball_json(gamma),'gamma_squared_defect':matrix_json(defect),'positive':check},gamma


def exact_frames(bases,tick=lambda:None):
    require(1<=len(bases)<=18,'cyclic frame count')
    n=bases[0].nrows();require(1<=n<=13,'frame dimension')
    frames=[];inverses=[];witness=[]
    for j,base in enumerate(bases):
        tick();require(base.nrows()==base.ncols()==n and all(v.is_exact() and v.is_finite() for row in base.tolist() for v in row),'fixed exact frame proposal')
        frame=base.transpose();inverse=frame.inv();defect=norm_inf_upper(eye(n)-midpoint_matrix(inverse)*frame)
        require(defect.is_finite() and defect<1,'fixed frame inverse')
        frames.append(frame);inverses.append(inverse);witness.append({'phase':j,'B':matrix_json(frame),'inverse':matrix_json(inverse),'inverse_defect':ball_json(defect)})
    return frames,inverses,witness


def advance(acc,phase):
    """Native phase polynomial times prior polynomial, plus scalar2 errors."""
    a,epsilon_a,gamma_a=phase;b,epsilon_b,gamma_b=acc;n=a[0].nrows()
    require(len(a)==len(b)==3 and epsilon_a>=0 and epsilon_b>=0 and gamma_a>=0 and gamma_b>=0,'positive error/operator bounds')
    full=[arb_mat(n,n) for _ in range(5)]
    for i in range(3):
        for j in range(3):full[i+j]+=a[i]*b[j]
    norm_a=gamma_a+frobenius_upper(a[1])+frobenius_upper(a[2])
    norm_b=gamma_b+frobenius_upper(b[1])+frobenius_upper(b[2])
    dropped=frobenius_upper(full[3])+frobenius_upper(full[4])
    epsilon=(norm_a*epsilon_b+epsilon_a*norm_b+epsilon_a*epsilon_b+dropped).upper()
    gamma=(gamma_a*gamma_b).upper()
    detail={'phase_polynomial_norm_upper':ball_json(norm_a),'prior_polynomial_norm_upper':ball_json(norm_b),
      'dropped_norm_upper':ball_json(dropped),'error_norm_upper':ball_json(epsilon),'center_product_norm_upper':ball_json(gamma)}
    return (full[:3],epsilon,gamma),detail


def scalar_remainder_metric(coefficients,epsilon,center_norm,tick=lambda:None):
    n=coefficients[0].nrows();point=midpoint_matrix(coefficients[0]);tick()
    proposal=solve_discrete_lyapunov(midpoint_float(point).T,np.eye(n),method='direct')
    p=exact_float_matrix((proposal+proposal.T)/2);pc=positive_definite(p,tick)
    d0=p-point.transpose()*p*point;dc=positive_definite(d0,tick)
    condition=norm_inf_upper(p)*norm_inf_upper(p.inv());defect=norm_inf_upper(d0-eye(n))
    guard=bool(condition.is_finite() and condition<2**40 and defect.is_finite() and defect<arb('1/100'))
    dcoeff=[arb_mat(n,n) for _ in range(5)];dcoeff[0]=p+arb_mat(n,n)
    for i in range(3):
        for j in range(3):dcoeff[i+j]-=coefficients[i].transpose()*p*coefficients[j]
    poly_bound=center_norm+frobenius_upper(coefficients[1])+frobenius_upper(coefficients[2])
    # Exact symmetric P has ||P||2 <= ||P||infinity. The scalar subtraction
    # dominates all Q^T P E, E^T P Q and E^T P E quadratic-form errors.
    pbound=norm_inf_upper(p);penalty=(pbound*(2*poly_bound*epsilon+epsilon*epsilon)).upper()
    complete=polynomial_value(dcoeff,arb(0,1))-eye(n)*penalty
    cc=positive_definite(complete,tick)
    return {'passed':bool(pc['passed'] and dc['passed'] and guard and cc['passed']),
      'P':matrix_json(p),'P_positive':pc,'point_D':matrix_json(d0),'point_D_positive':dc,
      'condition_inf_upper':ball_json(condition),'point_defect':ball_json(defect),'proposal_guard':guard,
      'D_coefficients':[matrix_json(v) for v in dcoeff],'polynomial_norm_upper':ball_json(poly_bound),
      'remainder_norm_upper':ball_json(epsilon),'P_norm_upper':ball_json(pbound),'remainder_penalty':ball_json(penalty),
      'complete_D':matrix_json(complete),'complete_D_positive':cc,
      'scope':'Conditional transformed monodromy attraction for this saved derivative domain only; exact cyclic similarity preserves native multipliers.'}


def framed_cocycle(rows,bases,halfwidth,*,tick=lambda:None,observe=lambda k,r:None):
    require(len(rows)==len(bases) and 1<=len(rows)<=18,'same complete native phase inventory')
    frames,inverses,proof=exact_frames(bases,tick);observe('frames',{'witnesses':proof,'cyclic_last_index':0})
    n=frames[0].nrows();acc=([eye(n),arb_mat(n,n),arb_mat(n,n)],arb(0),arb(1));phases=[]
    for j,row in enumerate(rows):
        require(row['phase']==j,'native phase order')
        original,er=phase_model(row['center_J'],row['whole_Jprime'],halfwidth)
        left=frames[(j+1)%len(rows)];right=inverses[j]
        coeff=[left*v*right for v in original];error=left*er*right
        observe('norm_phase_start',{'phase':j,'coefficients':[matrix_json(v) for v in coeff],'matrix_remainder':matrix_json(error)})
        op,gamma=operator_upper(coeff[0],tick);eps=frobenius_upper(error)
        record={'phase':j,'coefficients':[matrix_json(v) for v in coeff],'matrix_remainder':matrix_json(error),
          'remainder_norm_upper':ball_json(eps),'operator_bound':op}
        observe('norm_phase',record);require(op['passed'],'phase center operator bound')
        acc,detail=advance(acc,(coeff,eps,gamma));detail.update(phase=j,coefficients=[matrix_json(v) for v in acc[0]])
        observe('norm_product',detail);phases.append(record)
    metric=scalar_remainder_metric(*acc,tick=tick);observe('norm_metric',metric)
    return {'schema':'fixed-cyclic-frames-scalar-remainder-v1','passed':metric['passed'],'halfwidth':ball_json(halfwidth),
      'phases':len(rows),'coefficients':[matrix_json(v) for v in acc[0]],'remainder_norm_upper':ball_json(acc[1]),'metric':metric,
      'theorem':'B_j fixed/invertible, B_p=B_0; transformed native product is B_0 M B_0^-1. Every phase polynomial includes its unscaled center-root error and scaled uniform derivative-defect. Operator2 remainder recurrence retains all nonlinear and truncation terms. Strict positive P and complete decrease prove attraction on the inherited root subinterval.',
      'scope':'saved-derivative enclosure only; no whole branch, fold endpoint, boundary or utility claim'}
