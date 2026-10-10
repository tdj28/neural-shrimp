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
from .rigorous_arclength_slices import localize_slice
from .rigorous_connection import restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_orbits import exact, vector_json
from fractions import Fraction


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


def correlated_slice(model,jet_model,chart,whole_box,whole_tangent,left,right,*,
                     precision=256,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None):
    """One fixed interval; caller authenticates conditional root/tangent inputs."""
    lo,hi=Fraction(left),Fraction(right);require(lo<hi,'nonzero fixed slice')
    n=model.n;p=chart['statement']['period'];size=p*n+1
    require(jet_model.n==n and len(whole_box)==len(whole_tangent)==size,'full native root and total tangent')
    with ctx.workprec(precision):
        sc=(lo+hi)/2;delta=exact(str((hi-lo)/2))
        local=localize_slice(model,chart,str(sc),str(sc),iterations=3,precision=precision,
                             synthetic_profile=synthetic_profile,tick=tick,observe=observe)
        center_box=[restore_ball(v) for v in local['final_state_box']]
        domain=[restore_ball(v) for v in whole_box]
        tangent=[restore_ball(v) for v in whole_tangent]
        require(all(v.is_finite() for v in domain+tangent+center_box) and domain[-1]>0 and center_box[-1]>0,'finite positive native domains')
        observe('correlated_derivative_inputs',{'exact_endpoints':[str(lo),str(hi)],'center_parameter':str(sc),
          'actual_center_root_box':vector_json(center_box),'actual_whole_root_box':vector_json(domain),
          'actual_whole_total_tangent':vector_json(tangent),'halfwidth':ball_json(delta),
          'scope':'actual post-decoding native physical inputs; total tangent is a separately bound implicit enclosure'})
        models=[]
        for j in range(p):
            tick();_,j0,_=model.step(center_box[j*n:(j+1)*n],center_box[-1])
            tick();whole_j,derivative=total_jacobian_derivative(jet_model,domain[j*n:(j+1)*n],domain[-1],tangent[j*n:(j+1)*n],tangent[-1])
            phase=phase_model(j0,derivative,delta);models.append(phase)
            observe('correlated_phase_model',{'phase':j,'center_J':matrix_json(j0),
              'whole_J_mixed_diagnostic':matrix_json(whole_j),'whole_total_Jprime':matrix_json(derivative),
              'coefficients':[matrix_json(v) for v in phase[0]],'remainder':matrix_json(phase[1]),
              'theorem':'J(sc+delta*t)=A+B*t+E: E contains center_J-A plus integral of delta*Jprime-B over the segment from0 to t; no exact-derivative assumption for fixed slope B.'})
        product=cocycle(models,tick=tick,observe=observe)
        witness=monodromy_metric(product,tick=tick)
        observe('correlated_metric',witness)
        return {'schema':'conditional-correlated-gauge-slice-v1','passed':witness['passed'],
          'scientific_claim':not synthetic_profile,'exact_endpoints':[str(lo),str(hi)],
          'center_localization':local,'coefficients':[matrix_json(v) for v in product[0]],
          'remainder':matrix_json(product[1]),'metric':witness,
          'scope':'attraction only on this authenticated inherited primitive-root slice when all bounds pass; no whole-branch or neutral-fold attraction'}
