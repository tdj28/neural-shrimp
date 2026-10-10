"""Exploratory tracking and extrapolation of saved full-midpoint multipliers.

No spectral or bifurcation certification follows from these floating operations.
"""
from fractions import Fraction
import numpy as np
from scipy.optimize import linear_sum_assignment
from .rigorous_eigenpair_locus import require


def unpack(record):
    values=np.asarray(record['eigenvalues'],dtype=float)
    vectors=np.asarray(record['eigenvectors'],dtype=float)
    require(values.shape==(13,2) and vectors.shape==(13,13,2),'fixed native spectral dimensions')
    values=values[:,0]+1j*values[:,1];vectors=vectors[:,:,0]+1j*vectors[:,:,1]
    require(np.isfinite(values).all() and np.isfinite(vectors).all(),'finite saved spectrum')
    norm=np.linalg.norm(vectors,axis=0);require(np.all(norm>0),'nonzero eigenvectors')
    vectors=vectors/norm
    critical=record['critical_nearest_index'];require(type(critical)is int and 0<=critical<13,'critical index')
    residuals=np.asarray(record['relative_residuals'],dtype=float)
    require(residuals.shape==(13,) and np.isfinite(residuals).all() and np.max(residuals)<1e-8,'explicit midpoint eigen residual')
    require(abs(values[critical]-1)<1e-6 and float(record['mode_residual'])<1e-8,'authenticated mode numerical consistency')
    keep=[i for i in range(13)if i!=critical]
    return values[keep],vectors[:,keep],abs(values[critical]-1)


def assignment(previous_values,previous_vectors,values,vectors):
    distances=abs(previous_values[:,None]-values[None,:])/np.maximum(1,np.maximum(abs(previous_values[:,None]),abs(values[None,:])))
    overlaps=np.clip(abs(previous_vectors.conj().T@vectors),0,1)
    cost=distances+.2*(1-overlaps);rows,cols=linear_sum_assignment(cost)
    require(np.array_equal(rows,np.arange(len(values))),'complete spectral assignment')
    margins=[float(min(np.delete(cost[i],cols[i]))-cost[i,cols[i]]) for i in rows]
    return cols,{'maximum_assignment_cost':float(max(cost[rows,cols])),'minimum_row_assignment_margin':min(margins),'minimum_matched_overlap':float(min(overlaps[rows,cols])),'scope':'Floating eigenvalue/projective-overlap assignment; a nonpositive row margin is unresolved identity, not a mode-collision proof.'}


def future_linear_prediction(s,y,window,target):
    require(window in (4,8,16) and len(s)==len(y) and len(s)>=window,'fixed exploratory fit window')
    x=np.asarray(s[-window:],float);v=np.asarray(y[-window:],float)
    require(np.isfinite(x).all() and np.isfinite(v).all() and np.all(np.diff(x)>0),'finite ordered fit input')
    center=x[-1];a=np.column_stack([np.ones(window),x-center]);coef=np.linalg.lstsq(a,v,rcond=None)[0]
    residual=float(max(abs(a@coef-v)));slope=float(coef[1]);prediction=None if slope==0 else float(center+(target-coef[0])/slope)
    return {'window':window,'target':target,'slope':slope,'last_fitted_value':float(coef[0]),'maximum_fit_residual':residual,'future_gauge':prediction if prediction is not None and np.isfinite(prediction) and prediction>center else None,'scope':'Linear continuation proposal only, not a confidence interval or validated bracket.'}


def summarize(rows):
    require(len(rows)>=16,'retain complete predeclared chain')
    gauges=[float(Fraction(r['gauge']))for r in rows];require(np.all(np.diff(gauges)>0),'strictly increasing native gauge')
    history=[];output=[];previous=None
    for r in rows:
        values,vectors,critical_error=unpack(r['spectrum']);match=None
        if previous is not None:
            order,match=assignment(*previous,values,vectors);values=values[order];vectors=vectors[:,order]
        history.append(values);previous=values,vectors
        output.append({'gauge':r['gauge'],'rates':r['rates'],'values':[[float(v.real),float(v.imag)]for v in values],'critical_midpoint_error':float(critical_error),'matching':match,'complementary_radius':float(max(abs(values))),'distance_to_plus_one':float(min(abs(values-1))),'distance_to_minus_one':float(min(abs(values+1))),'distance_to_unit_circle':float(min(abs(abs(values)-1))),'fold_c_proxy':r['fold']['second_derivative_contraction'],'border_condition_proxy':r['fold']['border_frobenius_condition_proxy']})
    h=np.asarray(history);mode=int(np.argmin(abs(h[-1]-1)));real=bool(np.max(abs(h[:,mode].imag))<1e-8)
    predictions=[future_linear_prediction(gauges,h[:,mode].real,w,1.) for w in (4,8,16)]if real else []
    return {'scope':'Rung0 exploratory saved-midpoint trends. No new derivative or trajectory acquisition, no critical crossing proof, and no extrapolation used as a chart anchor.','rows':output,'nearest_plus_mode':mode,'nearest_plus_mode_real_throughout':real,'plus_one_linear_proposals':predictions,'last_mode':[[float(v.real),float(v.imag)]for v in h[:,mode]],'all_complements_numerically_inside_unit_circle':bool(np.all(abs(h)<1)),'negative_fold_c_proxy_throughout':all(r['fold_c_proxy']<0 for r in output)}
