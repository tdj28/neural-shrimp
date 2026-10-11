"""Saved full11 quotient error-budget diagnosis; hypothetical cases are not proofs."""
import numpy as np
from flint import arb,arb_mat
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import eye,midpoint_matrix,ball_json,matrix_json
from neural_shrimp.rigorous_periodic_lyapunov import midpoint_float
from neural_shrimp.rigorous_norm_cocycle import operator_upper,frobenius_upper,advance,scalar_remainder_metric

SCENARIOS=(('original',1,1,True,True),('center_error_floor',0,1,True,False),('remove_center_uncertainty',1,1,False,True),('remove_derivative_uncertainty',1,1,True,False),('remove_both_uncertainties',1,1,False,False),('same_center_halfwidth_1_2',1,2,True,True),('same_center_halfwidth_1_4',1,4,True,True))

def phase_parts(center,derivative,h):
    a=midpoint_matrix(center);hd=derivative*h;b=midpoint_matrix(hd)
    return [a,b,arb_mat(center.nrows(),center.ncols())],center-a,hd-b


def analyze(phases,h,*,tick=lambda:None,observe=lambda *_:None):
    require(len(phases)==18 and h.is_exact()and h>0,'literal18-phase positive-width diagnosis')
    require(all(a.nrows()==a.ncols()==b.nrows()==b.ncols()==11 and all(x.is_finite()for m in(a,b)for row in m.tolist()for x in row)for a,b in phases),'entire finite11-dimensional quotient')
    point=eye(11);components=[];gammas=[]
    for j,(q,qp)in enumerate(phases):
        coeff,e0,e1=phase_parts(q,qp,h);op,gamma=operator_upper(coeff[0],tick);require(op['passed'],'validated phase center norm')
        point=coeff[0]*point;gammas.append(gamma)
        components.append(dict(phase=j,center_error=ball_json(frobenius_upper(e0)),scaled_derivative_error=ball_json(frobenius_upper(e1)),center_operator=op))
    eig=np.linalg.eigvals(midpoint_float(point));require(np.isfinite(eig).all(),'finite numerical midpoint diagnostic')
    floating=dict(spectral_radius=float(max(abs(eig))),modes=sorted([dict(real=float(z.real),imag=float(z.imag),modulus=float(abs(z)))for z in eig],key=lambda v:v['modulus'],reverse=True),scope='Binary64 eigenvalues of the saved midpoint quotient product only; neither instability nor a whole-domain bound.')
    observe('phase-components',dict(rows=components,midpoint_product=matrix_json(point),numerical_midpoint=floating));answers=[]
    for name,num,den,center_on,derivative_on in SCENARIOS:
        width=h*num/den;acc=([eye(11),arb_mat(11,11),arb_mat(11,11)],arb(0),arb(1))
        for j,(q,qp)in enumerate(phases):
            coeff,e0,e1=phase_parts(q,qp,width);error=(e0 if center_on else arb_mat(11,11))+(e1*arb(0,1)if derivative_on else arb_mat(11,11));acc,_=advance(acc,(coeff,frobenius_upper(error),gammas[j]));tick()
        metric=scalar_remainder_metric(*acc,tick=tick)
        hypothetical=not(center_on and derivative_on)and num!=0
        row=dict(name=name,relative_halfwidth=f'{num}/{den}',metric_passed=metric['passed'],remainder=ball_json(acc[1]),penalty=metric['remainder_penalty'],metric_condition=metric['condition_inf_upper'],metric_point_defect=metric['point_defect'],metric_point_positive=metric['point_D_positive']['passed'],metric_positive=metric['P_positive']['passed'],metric_proposal_guard=metric['proposal_guard'],polynomial_norm=metric['polynomial_norm_upper'],hypothetical=hypothetical,center_uncertainty_included=center_on,derivative_uncertainty_included=derivative_on)
        observe(name,dict(summary=row,metric=metric));answers.append(row)
    return dict(scenarios=answers,numerical_midpoint=floating,scope='Diagnosis of one fixed saved center under the original derivative domain. Removed uncertainty is hypothetical; narrower centered scenarios do not cover the failed interval or neighboring centers. No new stability or instability conclusion.')
