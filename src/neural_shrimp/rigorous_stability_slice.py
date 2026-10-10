"""Fresh known-root slice, implicit tangent and fixed-frame cocycle witness.

This composes existing conditional theorems. It never establishes a new root:
existence, primitive period and branch identity are inherited from the chart.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_orbits import exact,eye,matrix_json,vector_json,ball_json
from .rigorous_arclength_slices import localize_slice
from .rigorous_correlated_cocycle import total_jacobian_derivative
from .rigorous_periodic_lyapunov import propose_metrics,positive_definite
from .rigorous_norm_cocycle import framed_cocycle


def uniform_tangent(pre,jac,weights):
    """A U'=e_last via a strict weighted Neumann bound and three intersections."""
    n=jac.nrows();require(jac.ncols()==pre.nrows()==pre.ncols()==len(weights)==n,'square implicit system')
    require(all(w.is_exact() and w>0 for w in weights),'exact positive derivative weights')
    d=eye(n)-pre*jac
    def vn(x):return max((abs(x[i,0]).upper()/weights[i]).upper() for i in range(n))
    q=max((sum((abs(d[i,j]).upper()*weights[j] for j in range(n)),arb(0))/weights[i]).upper() for i in range(n))
    result={'weighted_neumann_bound':ball_json(q),'passed':bool(q.is_finite() and q<1)}
    if not result['passed']:return result
    v=pre*arb_mat([[int(i==n-1)] for i in range(n)])
    radius=(vn(d*v)/(1-q)).upper()
    y=arb_mat([[arb(v[i,0],weights[i]*radius)] for i in range(n)])
    result.update(preconditioned_rhs=matrix_json(v),initial_derivative_box=matrix_json(y),weighted_error_bound=ball_json(radius))
    rows=[]
    for _ in range(3):
        image=v+d*y;updated=arb_mat([[y[i,0].intersection(image[i,0])] for i in range(n)])
        rows.append({'image':matrix_json(image),'retained':matrix_json(updated)});y=updated
    result.update(refinements=rows,derivative_box=matrix_json(y),
      scope='Uniform derivative of the inherited root in the same exact gauge on the whole closed slice; the final localization derivative domain contains every exact root.')
    return result


def stability_slice(model,jet_model,chart,left,right,*,precision=256,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None):
    lo,hi=Fraction(left),Fraction(right);require(lo<hi,'nonzero exact slice')
    p=chart['statement']['period'];n=model.n;size=p*n+1
    require(jet_model.n==n and len(chart['exact_weights'])==size,'same native model and full derivative weights')
    def observed(prefix):return lambda kind,row:observe(prefix+'_'+kind,row)
    with ctx.workprec(precision):
        whole=localize_slice(model,chart,str(lo),str(hi),precision=precision,synthetic_profile=synthetic_profile,tick=tick,observe=observed('whole'))
        final=whole['intersections'][-1]
        tangent=uniform_tangent(restore_matrix(whole['start']['preconditioner']),restore_matrix(final['jacobian']),[exact(v) for v in chart['exact_weights']])
        observe('uniform_tangent',{'exact_endpoints':[str(lo),str(hi)],'bound':tangent,
          'root_derivative_domain':final['derivative_domain'],'root_final_state_box':whole['final_state_box']})
        whole.pop('start');whole.pop('intersections')
        if not tangent['passed']:
            return {'passed':False,'status':'inconclusive_uniform_tangent','whole_localization':whole,'uniform_tangent':tangent,'exact_endpoints':[str(lo),str(hi)]}
        sc=(lo+hi)/2;delta=exact(str((hi-lo)/2))
        center=localize_slice(model,chart,str(sc),str(sc),precision=precision,synthetic_profile=synthetic_profile,tick=tick,observe=observed('center'))
        center.pop('start');center.pop('intersections')
        pointbox=[restore_ball(v) for v in center['final_state_box']]
        box=[restore_ball(v) for v in whole['final_state_box']]
        derivative=restore_matrix(tangent['derivative_box']);velocity=[derivative[i,0] for i in range(size)]
        require(len(pointbox)==len(box)==size and all(v.is_finite() for v in pointbox+box+velocity) and pointbox[-1]>0 and box[-1]>0,'finite actual derivative domains')
        observe('physical_inputs',{'exact_endpoints':[str(lo),str(hi)],'center_parameter':str(sc),'actual_center_box':vector_json(pointbox),
          'actual_whole_box':vector_json(box),'actual_uniform_tangent':vector_json(velocity),'halfwidth':ball_json(delta)})
        rows=[];centers=[]
        for j in range(p):
            tick();_,jac,_=model.step(pointbox[j*n:(j+1)*n],pointbox[-1]);centers.append(jac)
            tick();mixed,prime=total_jacobian_derivative(jet_model,box[j*n:(j+1)*n],box[-1],velocity[j*n:(j+1)*n],velocity[-1])
            row={'phase':j,'center_J':jac,'whole_Jprime':prime};rows.append(row)
            observe('phase_derivatives',{'phase':j,'center_J':matrix_json(jac),'whole_J_mixed_diagnostic':matrix_json(mixed),'whole_total_Jprime':matrix_json(prime)})
        # All metrics/bases here are proposals. The norm cocycle independently
        # certifies every exact frame inverse, operator bound and final decrease.
        tick();metrics=propose_metrics(centers);bases=[];checks=[]
        for j,metric in enumerate(metrics):
            check=positive_definite(metric,tick);checks.append({'phase':j,'P':matrix_json(metric),'positive':check})
            if check['passed']:bases.append(restore_matrix(check['basis']))
        observe('frame_proposals',{'phases':checks,'all_positive':len(bases)==p})
        result={'whole_localization':whole,'center_localization':center,'uniform_tangent':tangent,'exact_endpoints':[str(lo),str(hi)],
          'scope':'Full-state attraction only on this inherited primitive-root closed gauge slice if every strict gate passes; no neutral fold or whole-branch conclusion.'}
        if len(bases)!=p:return dict(result,passed=False,status='inconclusive_frame_proposal')
        witness=framed_cocycle(rows,bases,delta,tick=tick,observe=observe)
        return dict(result,passed=witness['passed'],status='complete_slice_bounds',norm_cocycle=witness)
