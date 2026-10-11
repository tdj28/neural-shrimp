"""Saved-ball width-scaling proposals, never a proof on an enlarged domain."""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,eye,matrix_json,vector_json,ball_json
from .rigorous_eigenpair_locus import require,column,finite
from .rigorous_eigenpair_march import decode,compact


def split_width(chart,observe=lambda k,v:None):
    compact(chart)
    with ctx.workprec(chart['settings']['precision']):
        size=chart['shooting_dimension'];delta=exact(str(Fraction(2)**chart['settings']['halfwidth_exponent']))
        mat=lambda key:arb_mat([[decode(v) for v in row] for row in chart[key]])
        C=mat('preconditioner');J0=mat('center_jacobian');D=mat('defect');coeff=mat('quadratic_center_coefficients');whole=mat('quadratic_whole_coefficients')
        radii=[decode(v) for v in chart['coordinate_radii']]
        require(C.nrows()==C.ncols()==J0.nrows()==J0.ncols()==D.nrows()==D.ncols()==size and coeff.nrows()==whole.nrows()==size and coeff.ncols()==whole.ncols()==4,'saved dimensions')
        require(all(v.is_finite() and v>0 for v in radii) and all(finite(v) for v in (C,J0,D,coeff,whole)),'finite saved data')
        D0=eye(size)-C*J0;observe('center_defect_return',{'matrix':matrix_json(D0)})
        actions=[]
        for degree in range(4):
            source=coeff if degree<3 else whole
            action=-C*column([source[i,degree] for i in range(size)])
            observe('coefficient_action_return',{'degree':degree,'action':matrix_json(action)})
            actions.append(action)
        q0=[];qv=[];qold=[];forcing=[[] for _ in range(4)];one=[];two=[]
        powers=[arb(1),delta,delta*delta,delta*delta*delta]
        for i in range(size):
            base=sum((abs(D0[i,j]).upper()*radii[j] for j in range(size)),arb(0))/radii[i]
            variation=sum((abs(D[i,j]-D0[i,j]).upper()*radii[j] for j in range(size)),arb(0))/radii[i]
            original=sum((abs(D[i,j]).upper()*radii[j] for j in range(size)),arb(0))/radii[i]
            term=[abs(actions[k][i,0]).upper()*powers[k]/radii[i] for k in range(4)]
            q0.append(base);qv.append(variation);qold.append(original)
            for k in range(4):forcing[k].append(term[k])
            one.append(base+variation+sum(term,arb(0)))
            # Heuristic doubles the *observed* domain-dependent defect. It is
            # neither an enclosure nor a Lipschitz assertion on the wider box.
            two.append(base+2*variation+sum((2**k*term[k] for k in range(4)),arb(0)))
        maximum=lambda vs:max(v.upper() for v in vs)
        result={'schema':'neural-shrimp-eigenpair-saved-width-budget-v1','old_halfwidth_exponent':chart['settings']['halfwidth_exponent'],'proposed_halfwidth_exponent':chart['settings']['halfwidth_exponent']+1,
          'center_defect_row_bounds':vector_json(q0),'observed_variation_row_bounds':vector_json(qv),'whole_defect_row_bounds':vector_json(qold),'forcing_degree_row_bounds':[vector_json(v) for v in forcing],
          'same_width_triangle_proxy_rows':vector_json(one),'double_width_proxy_rows':vector_json(two),
          'summary':{'center_defect_max':ball_json(maximum(q0)),'observed_variation_max':ball_json(maximum(qv)),'whole_defect_max':ball_json(maximum(qold)),'forcing_degree_max':[ball_json(maximum(v)) for v in forcing],'same_width_triangle_proxy_max':ball_json(maximum(one)),'double_width_proxy_max':ball_json(maximum(two)),'double_width_proxy_below_one':bool(maximum(two)<1)},
          'scope':'Saved-array proposal only. Widened derivatives, path coefficients, predictor and preconditioner can change; no new domain is enclosed and no proof gate is passed by these proxies. A wider pilot must recompute all native C1/path returns and strict root/primitive/full-anchor gates.'}
        observe('width_budget_return',result);return result
