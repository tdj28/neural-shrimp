"""A common scalar variable through native phase matrices, with all remainders.

Fixed coefficient proposals need not equal exact derivatives. The integral
remainder explicitly includes their defect. No floating operation is a gate.
"""
from flint import arb, arb_mat, ctx
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from .rigorous_connection import require
from .rigorous_orbits import eye, midpoint_matrix, norm_inf_upper, matrix_json, ball_json
from .rigorous_folds import MixedJet
from .rigorous_periodic_lyapunov import midpoint_float, exact_float_matrix, positive_definite


def integer_power(variable, exponent):
    # Arb's generic ** path uses a real-power extension and may reject balls
    # spanning negative values. Integer polynomial powers use multiplication.
    require(type(exponent) is int and 0 <= exponent <= 4,'fixed integer power')
    if exponent and exponent % 2 == 0 and variable.contains(0):
        bound=abs(variable).upper();value=arb(1)
        for _ in range(exponent):value*=bound
        return arb(0).union(value.upper())
    value=arb(1)
    for _ in range(exponent):value*=variable
    return value


def polynomial_value(coefficients, variable):
    n=coefficients[0].nrows()
    result=arb_mat(n,n)
    for k,value in enumerate(coefficients):
        result += value*integer_power(variable,k)
    return result


def multiply_models(left,right,degree=2):
    """Noncommuting left*right, retaining one common t in [-1,1]."""
    require(degree==2,'fixed quadratic truncation')
    a,ea=left;b,eb=right;n=a[0].nrows();t=arb(0,1)
    require(len(a)==len(b)==3 and all(v.nrows()==v.ncols()==n for v in a+b+[ea,eb]),'square coefficient inventory')
    full=[arb_mat(n,n) for _ in range(5)]
    for i in range(3):
        for j in range(3):full[i+j]+=a[i]*b[j]
    remainder=ea*polynomial_value(b,t)+polynomial_value(a,t)*eb+ea*eb
    for k in range(3,5):remainder+=full[k]*integer_power(t,k)
    return full[:3],remainder


def total_jacobian_derivative(jet_model,theta,rate,theta_prime,rate_prime):
    """Physical DF_theta and its total derivative along the root branch.

    The physical base gradient holds rate fixed; the directional derivative
    includes the actual state tangent and actual hidden-rate tangent.
    """
    n=jet_model.n
    require(len(theta)==len(theta_prime)==n,'native state tangent dimension')
    variables=[MixedJet.variable(theta[i],n,i,theta_prime[i]) for i in range(n)]
    parameter=MixedJet(rate,[arb(0)]*n,rate_prime,[arb(0)]*n)
    result=jet_model.jet_step(variables,parameter)
    require(len(result)==n,'native output dimension')
    jac=arb_mat([[v.g[i] for i in range(n)] for v in result])
    derivative=arb_mat([[v.m[i] for i in range(n)] for v in result])
    require(all(v.is_finite() for matrix in (jac,derivative) for row in matrix.tolist() for v in row),'finite mixed derivative')
    return jac,derivative


def phase_model(center_jac,whole_jac_derivative,halfwidth):
    """J(s0+delta*t)=A+B*t+E, from the integral derivative identity."""
    require(halfwidth.is_exact() and halfwidth.is_finite() and halfwidth>0,'exact positive halfwidth')
    a=midpoint_matrix(center_jac)
    derivative=whole_jac_derivative*halfwidth
    b=midpoint_matrix(derivative);t=arb(0,1);n=a.nrows()
    error=(center_jac-a)+(derivative-b)*t
    return [a,b,arb_mat(n,n)],error


def cocycle(models,*,tick=lambda:None,observe=lambda k,r:None):
    require(1<=len(models)<=18,'fixed phase cap')
    n=models[0][0][0].nrows();require(1<=n<=13,'native dimension cap')
    accumulated=([eye(n),arb_mat(n,n),arb_mat(n,n)],arb_mat(n,n))
    for j,model in enumerate(models):
        tick();accumulated=multiply_models(model,accumulated)
        observe('correlated_product',{'phase':j,'coefficients':[matrix_json(v) for v in accumulated[0]],
                                     'remainder':matrix_json(accumulated[1])})
    return accumulated


def monodromy_metric(model,*,tick=lambda:None):
    coefficients,error=model;n=coefficients[0].nrows();t=arb(0,1)
    require(len(coefficients)==3 and all(v.nrows()==v.ncols()==n for v in coefficients+[error]),'square complete model')
    point=midpoint_matrix(coefficients[0]);floating=midpoint_float(point)
    tick();proposed=solve_discrete_lyapunov(floating.T,np.eye(n),method='direct')
    p=exact_float_matrix((proposed+proposed.T)/2)
    pcheck=positive_definite(p,tick)
    d0=p-point.transpose()*p*point;pointcheck=positive_definite(d0,tick)
    condition=norm_inf_upper(p)*norm_inf_upper(p.inv())
    defect=norm_inf_upper(d0-eye(n))
    guard=bool(condition.is_finite() and condition<2**40 and defect.is_finite() and defect<arb('1/100'))
    # Keep shared powers until after all phase and left/right cancellations.
    dcoeff=[arb_mat(n,n) for _ in range(5)];dcoeff[0]=p+arb_mat(n,n)
    for i in range(3):
        for j in range(3):dcoeff[i+j]-=coefficients[i].transpose()*p*coefficients[j]
    polynomial=polynomial_value(coefficients,t)
    perturbation=-polynomial.transpose()*p*error-error.transpose()*p*polynomial-error.transpose()*p*error
    decrease=polynomial_value(dcoeff,t)+perturbation
    intervalcheck=positive_definite(decrease,tick)
    return {'schema':'common-gauge-monodromy-metric-v1',
      'passed':bool(pcheck['passed'] and pointcheck['passed'] and guard and intervalcheck['passed']),
      'P':matrix_json(p),'P_positive':pcheck,'point_D':matrix_json(d0),'point_D_positive':pointcheck,
      'condition_inf_upper':ball_json(condition),'point_defect_from_identity':ball_json(defect),'proposal_guard':guard,
      'D_coefficients':[matrix_json(v) for v in dcoeff],'remainder_perturbation':matrix_json(perturbation),
      'interval_D':matrix_json(decrease),'interval_D_positive':intervalcheck,
      'theorem':'The complete native monodromy lies in the saved polynomial plus remainder at the same t in [-1,1]. Strict P>0 and P-M(t)^T P M(t)>0 give full-state attraction for every enclosed root. All omitted product terms and polynomial/remainder cross terms are included.',
      'scope':'conditional on authenticated root, native phase order and total phase-Jacobian derivative enclosures; no endpoint-neutral attraction or whole-boundary claim'}
