"""Containment refinement of an already proved parameter-dependent root.

This module does not establish existence. For each parameter in a parent tube,
the parent theorem supplies a unique root and a correction enclosure. Every
iteration below retains that root by a centered mean-value/Krawczyk identity.
"""
from fractions import Fraction
from flint import arb, arb_mat, ctx
from .rigorous_orbits import exact, eye, ball_json, matrix_json, norm_inf_upper
from .rigorous_connection import require, restore_ball, shooting
from .rigorous_connection_frames import restore_matrix
from .rigorous_periodic_lyapunov import periodic_witness


def finite_matrix(matrix):
    return all(matrix[i,j].is_finite() for i in range(matrix.nrows()) for j in range(matrix.ncols()))


def localize_root(model, chart, left, right, *, iterations=3, precision=256,
                  tick=lambda:None, observe=lambda k,r:None):
    """Restrict, recenter, then intersect known E with -C R + (I-C A) E.

    For p(h)=u+s(h-hc), R encloses H(p(h),h) on the selected interval.
    Both the predictor path and the segments p(h)+t*e must lie in the derivative
    domain, hence hull(0,E), even when E does not contain zero. Root existence,
    primitive period and branch identity are inherited, never inferred here.
    """
    require(chart['gates']['uniform_root'], 'proved parent root required')
    require(type(iterations) is int and 1<=iterations<=3, 'fixed refinement cap')
    require(type(precision) is int and 128<=precision<=512, 'precision cap')
    lo,hi=Fraction(left),Fraction(right);a,b=map(Fraction,chart['exact_parameter_endpoints'])
    require(a<=lo<=hi<=b,'closed subinterval of parent required')
    p=chart['statement']['period'];n=model.n;dim=p*n
    require(1<=p<=18 and 1<=n<=13 and chart['statement']['phase_dimension']==n,'dimension cap and identity')
    with ctx.workprec(precision):
        hc=(lo+hi)/2;half=(hi-lo)/2;h0=Fraction(chart['parameter_center'])
        rate=arb(exact(str(hc)),exact(str(half)));d=arb(0,exact(str(half)))
        base=[restore_ball(v) for row in chart['center'] for v in row]
        slope=restore_matrix(chart['predictor_slope']);pre=restore_matrix(chart['preconditioner'])
        known=restore_matrix(chart['image_correction'])
        require(len(base)==dim and slope.nrows()==known.nrows()==dim and slope.ncols()==known.ncols()==1,'predictor dimensions')
        require(pre.nrows()==pre.ncols()==dim,'preconditioner dimensions')
        require(all(v.is_exact() and v.is_finite() for v in base) and finite_matrix(slope) and finite_matrix(pre) and finite_matrix(known),'finite parent algebra')
        require(all(slope[i,0].is_exact() for i in range(dim)) and all(pre[i,j].is_exact() for i in range(dim) for j in range(dim)),'fixed exact dyadic predictor and C')
        shifted=[base[i]+slope[i,0]*exact(str(hc-h0)) for i in range(dim)]
        center=[(shifted[i]+known[i,0].mid()).mid() for i in range(dim)]
        # Includes all arithmetic displacement; the rounded center is only a proposal.
        error=arb_mat([[shifted[i]+known[i,0]-center[i]] for i in range(dim)])
        initial_error=matrix_json(error)
        hcenter,_,_,_=shooting(model,[center[j*n:(j+1)*n] for j in range(p)],exact(str(hc)),tick)
        rows=[]
        observe('localization_start',{'exact_endpoints':[str(lo),str(hi)],'center_parameter':str(hc),
            'center':matrix_json(arb_mat([[v] for v in center])),'initial_correction':initial_error,
            'center_residual':matrix_json(hcenter)})
        for iteration in range(iterations):
            segment=[arb(0).union(error[i,0]) for i in range(dim)]
            boxes=[center[i]+slope[i,0]*d+segment[i] for i in range(dim)]
            _,jac,partial,_=shooting(model,[boxes[j*n:(j+1)*n] for j in range(p)],rate,tick)
            path=jac*slope+partial;residual=hcenter+path*d
            defect=eye(dim)-pre*jac;image=-pre*residual+defect*error
            require(finite_matrix(image),'finite refinement image')
            # Arb intersection can round outward: the saved result remains a
            # valid enclosure even when it is microscopically wider than E.
            updated=arb_mat([[error[i,0].intersection(image[i,0])] for i in range(dim)])
            row={'iteration':iteration,'prior_correction':matrix_json(error),
                 'predictor_error_segment':matrix_json(arb_mat([[v] for v in segment])),
                 'derivative_phase_boxes':[[ball_json(v) for v in boxes[j*n:(j+1)*n]] for j in range(p)],
                 'jacobian':matrix_json(jac),'parameter_partial':matrix_json(partial),
                 'path_derivative':matrix_json(path),'centered_residual':matrix_json(residual),
                 'defect':matrix_json(defect),'contraction_bound_diagnostic':ball_json(norm_inf_upper(defect)),
                 'image':matrix_json(image),'retained_correction':matrix_json(updated),
                 'max_correction_radius':ball_json(max(updated[i,0].rad().upper() for i in range(dim))),
                 'containment_identity_applied':True}
            observe('refinement',row);rows.append(row);error=updated
        boxes=[center[i]+slope[i,0]*d+error[i,0] for i in range(dim)]
        return {'schema':'neural-shrimp-known-root-localization-v1','passed':True,
            'exact_endpoints':[str(lo),str(hi)],'floor_only':lo==hi,'precision':precision,'iterations':iterations,
            'parameter_interval':ball_json(rate),'center_parameter':str(hc),
            'center':matrix_json(arb_mat([[v] for v in center])),'predictor_slope':matrix_json(slope),
            'preconditioner':matrix_json(pre),'initial_correction':initial_error,'center_residual':matrix_json(hcenter),
            'refinements':rows,'final_correction':matrix_json(error),
            'phase_boxes':[[ball_json(v) for v in boxes[j*n:(j+1)*n]] for j in range(p)],
            'inherited_primitive_period':bool(chart['gates']['primitive_period']),
            'scope':'conditional containment refinement of the parent unique root, with unchanged native phases; primitive period only when inherited from parent; no new existence or join theorem',
            'theorem':'For each fixed h the known root error e is in E and satisfies e=-C*H(p(h),h)+(I-C*A_h)e; A_h is enclosed on predictor+conv(0,E). Therefore e is in E intersect K(E). Repetition retains that same root.'}


def localized_stability(model, chart, left, right, *, iterations=3, precision=256,
                        tick=lambda:None, observe=lambda k,r:None):
    with ctx.workprec(precision):
        localized=localize_root(model,chart,left,right,iterations=iterations,precision=precision,tick=tick,observe=observe)
        rate=restore_ball(localized['parameter_interval']);blocks=[]
        for row in localized['phase_boxes']:
            tick();_,jac,_=model.step([restore_ball(v) for v in row],rate);blocks.append(jac)
        observe('localized_derivatives',{'phase_jacobians':[matrix_json(m) for m in blocks]})
        proof=periodic_witness(blocks,precision=precision,tick=tick,observe=observe)
        return {'localization':localized,'stability':proof,'passed':proof['passed'],
            'scope':'same parent root is attracting on this subinterval if the cyclic metric gates all pass; no untested interval coverage'}
