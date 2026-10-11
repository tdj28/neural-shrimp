"""Containment-only localization of a known root at an exact old gauge.

Existence, native phase identity and primitive period are inherited from the
parent chart. No contraction is required to intersect a known root enclosure
with its rigorous Krawczyk image. This does not extend the parameter interval.
"""
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,eye,midpoint_matrix,ball_json,matrix_json,vector_json,norm_inf_upper
from .rigorous_connection import require,restore_ball
from .rigorous_arclength_connection import bordered


def localize_endpoint(model,chart,*,iterations=3,precision=256,synthetic_profile=False,
                      tick=lambda:None,observe=lambda k,r:None):
    require(chart['passed'] and all(chart['gates'].values()) or synthetic_profile,'qualified inherited root')
    require(type(iterations) is int and 1<=iterations<=3,'fixed intersection cap')
    require(type(precision) is int and 128<=precision<=512,'precision cap')
    period=chart['statement']['period'];n=model.n;size=period*n+1
    require(1<=period<=18 and 1<=n<=13 and chart['statement']['phase_dimension']==n,'native dimensions')
    with ctx.workprec(precision):
        box=[restore_ball(v) for v in chart['endpoint_root_enclosures'][1]]
        normal=[exact(v) for v in chart['normal']];origin=[exact(v) for v in chart['origin']]
        parameter=exact(chart['exact_parameter_endpoints'][1])
        require(len(box)==len(normal)==len(origin)==size,'full augmented known root')
        require(all(v.is_exact() and v.is_finite() for v in normal+origin+[parameter]),'exact old gauge')
        require(all(v.is_finite() for v in box) and box[-1]>0,'finite positive-rate old box')
        center=[v.mid() for v in box];error=arb_mat([[box[i]-center[i]] for i in range(size)])
        initial=matrix_json(error)
        residual,jac0,blocks0=bordered(model,center,period,normal,origin,parameter,tick)
        tick();pre=midpoint_matrix(jac0.inv())
        start={'old_parameter':chart['exact_parameter_endpoints'][1],'old_normal':chart['normal'],'old_origin':chart['origin'],
          'initial_box':vector_json(box),'center':vector_json(center),'initial_error':initial,'center_residual':matrix_json(residual),
          'center_jacobian':matrix_json(jac0),'center_phase_jacobians':[matrix_json(v) for v in blocks0],'preconditioner':matrix_json(pre)}
        observe('localization_start',start);rows=[]
        for iteration in range(iterations):
            segment=[arb(0).union(error[i,0]) for i in range(size)]
            domain=[center[i]+segment[i] for i in range(size)]
            require(domain[-1]>0,'positive whole localization rate domain')
            _,jac,blocks=bordered(model,domain,period,normal,origin,parameter,tick)
            defect=eye(size)-pre*jac;image=-pre*residual+defect*error
            require(all(image[i,0].is_finite() for i in range(size)),'finite containment image')
            updated=arb_mat([[error[i,0].intersection(image[i,0])] for i in range(size)])
            row={'iteration':iteration,'prior_error':matrix_json(error),'derivative_domain':vector_json(domain),
              'jacobian':matrix_json(jac),'phase_jacobians':[matrix_json(v) for v in blocks],
              'defect':matrix_json(defect),'image':matrix_json(image),'retained_error':matrix_json(updated),
              'max_radius':ball_json(max(updated[i,0].rad().upper() for i in range(size))),
              'unweighted_contraction_diagnostic':ball_json(norm_inf_upper(defect))}
            observe('intersection',row);rows.append(row);error=updated
        final=[center[i]+error[i,0] for i in range(size)]
        return {'schema':'neural-shrimp-known-arclength-endpoint-localization-v1','passed':True,
          'scientific_claim':not synthetic_profile,'precision':precision,'iterations':iterations,
          'statement':chart['statement'],'parent_gates':chart['gates'],'start':start,'intersections':rows,
          'final_state_box':vector_json(final),'final_max_radius':ball_json(max(v.rad().upper() for v in final)),
          'scope':'same inherited native primitive root at exact old gauge; no new interval, attraction or fold membership',
          'theorem':'The inherited root error e lies in E and satisfies e=-C*G(center)+(I-C*A)e; the mean-value derivative lies in J(center+conv(0,E)). Intersect E with this image, retaining that root. Repeat without assuming contraction.'}
