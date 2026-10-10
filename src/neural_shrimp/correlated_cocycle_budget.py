"""Saved-only decomposition of one failed cocycle witness; no new derivatives."""
from flint import arb,arb_mat
from .rigorous_orbits import eye,norm_inf_upper,ball_json,matrix_json
from .rigorous_correlated_cocycle import polynomial_value,multiply_models
from .rigorous_connection import require


def gershgorin(matrix):
    n=matrix.nrows();require(matrix.ncols()==n,'square diagnostic')
    margins=[matrix[i,i].lower()-sum((abs(matrix[i,j]).upper() for j in range(n) if j!=i),arb(0)) for i in range(n)]
    return {'passed':all(v.is_finite() and v>0 for v in margins),'margins':[ball_json(v) for v in margins]}


def decompose(coefficients,remainder,p,inverse,phase_rows,tick=lambda:None):
    n=p.nrows();require(1<=n<=13 and p.ncols()==inverse.nrows()==inverse.ncols()==n,'native metric shape')
    require(len(coefficients)==3 and 1<=len(phase_rows)<=18,'fixed product inventory')
    t=arb(0,1);dcoeff=[arb_mat(n,n) for _ in range(5)];dcoeff[0]=p+arb_mat(n,n)
    for i in range(3):
        for j in range(3):dcoeff[i+j]-=coefficients[i].transpose()*p*coefficients[j]
    poly=polynomial_value(coefficients,t)
    residual_parts={'left':-poly.transpose()*p*remainder,'right':-remainder.transpose()*p*poly,'quadratic':-remainder.transpose()*p*remainder}
    def whiten(x):return inverse*x*inverse.transpose()
    def bound(x):return ball_json(norm_inf_upper(whiten(x)))
    poly_decrease=polynomial_value(dcoeff,t)
    full=poly_decrease+sum(residual_parts.values(),arb_mat(n,n))
    # Remove phase remainders only as a diagnostic counterfactual. Truncation
    # remains enclosed, but this is NOT the true empirical cocycle.
    truncated=([eye(n),arb_mat(n,n),arb_mat(n,n)],arb_mat(n,n));phases=[]
    for row in phase_rows:
        tick();cs=row['coefficients'];er=row['remainder']
        require(len(cs)==3 and all(m.nrows()==m.ncols()==n for m in cs+[er]),'phase dimensions')
        truncated=multiply_models((cs,arb_mat(n,n)),truncated)
        phases.append({'phase':row['phase'],'A_inf':ball_json(norm_inf_upper(cs[0])),'B_inf':ball_json(norm_inf_upper(cs[1])),
          'phase_remainder_inf':ball_json(norm_inf_upper(er)),
          'center_J_uncertainty_inf':ball_json(norm_inf_upper(row['center_J']-cs[0])),
          'scaled_derivative_variation_inf':ball_json(norm_inf_upper(row['scaled_Jprime']-cs[1]))})
    return {'schema':'saved-correlated-cocycle-budget-v1','phases':phases,
      'whitened_polynomial_coefficient_norms':[bound(v) for v in dcoeff],
      'whitened_remainder_terms':{name:bound(value) for name,value in residual_parts.items()},
      'whitened_polynomial_only_check':gershgorin(whiten(poly_decrease)),
      'whitened_full_check':gershgorin(whiten(full)),
      'saved_product_remainder_inf':ball_json(norm_inf_upper(remainder)),
      'zero_phase_error_counterfactual_truncation_inf':ball_json(norm_inf_upper(truncated[1])),
      'counterfactual_scope':'Phase remainder removal is a diagnostic only, not an enclosure of the actual cocycle.',
      'scope':'Saved matrix arithmetic and a sufficient-test decomposition only; no new model, derivative, inverse, eigensolver, factorization or attraction claim.'}
