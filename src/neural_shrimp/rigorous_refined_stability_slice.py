"""Compose reviewed slice, true-tangent refinement and native norm cocycle."""
from fractions import Fraction
from flint import ctx
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_orbits import exact,matrix_json,vector_json
from .rigorous_stability_slice import stability_slice
from .rigorous_tangent_variation import refine_tangent
from .rigorous_correlated_cocycle import total_jacobian_derivative
from .rigorous_norm_cocycle import framed_cocycle


def refined_stability_slice(model,first_jet,refined_jet,chart,left,right,*,precision=256,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None):
    """All repeated stages are intentional and declared; no old proof is edited.

    The first norm attempt is retained even if it fails. Only failed derivative
    or frame prerequisites stop before the second norm attempt. U'' is used on
    the SAME whole state/rate enclosure; both mixed models share the exact map.
    """
    held={};phases=[]
    def capture(kind,row):
        # Durable external observer precedes retaining a prerequisite in memory.
        observe('base_'+kind,row)
        if kind in ('whole_slice_start','whole_slice_intersection','center_slice_intersection','physical_inputs','frames'):held[kind]=row
        if kind=='phase_derivatives':phases.append(row)
    with ctx.workprec(precision):
        base=stability_slice(model,first_jet,chart,left,right,precision=precision,synthetic_profile=synthetic_profile,tick=tick,observe=capture)
        result={'passed':False,'base':base,'exact_endpoints':[str(Fraction(left)),str(Fraction(right))]}
        if base['status']!='complete_slice_bounds':return dict(result,status='inconclusive_base_prerequisites')
        p=chart['statement']['period'];n=model.n;size=p*n+1
        require(first_jet.n==refined_jet.n==n and first_jet.hidden==refined_jet.hidden==model.hidden,'same native dimensions/hidden prefix')
        require(len(phases)==p and [r['phase'] for r in phases]==list(range(p)),'complete native first derivatives')
        whole=held['whole_slice_intersection'];center=held['center_slice_intersection'];physical=held['physical_inputs'];start=held['whole_slice_start'];frames=held['frames']
        require(whole['iteration']==center['iteration']==2,'three complete intersections per domain')
        require(physical['exact_endpoints']==result['exact_endpoints'] and start['preconditioner']==chart['preconditioner'],'same gauge/preconditioner')
        velocity=restore_matrix(base['uniform_tangent']['derivative_box']);box=[restore_ball(v) for v in physical['actual_whole_box']]
        require(len(box)==size and vector_json([velocity[i,0] for i in range(size)])==physical['actual_uniform_tangent'],'same whole domain/tangent')
        delta=exact(str((Fraction(right)-Fraction(left))/2))
        refinement=refine_tangent(restore_matrix(start['preconditioner']),restore_matrix(whole['jacobian']),restore_matrix(center['jacobian']),
          [restore_matrix(r['whole_J_mixed_diagnostic']) for r in phases],[restore_matrix(r['whole_total_Jprime']) for r in phases],
          velocity,box[-1],model.hidden,[exact(v) for v in chart['exact_weights']],delta)
        observe('tangent_refinement',refinement);result['tangent_refinement']=refinement
        if not refinement['passed']:return dict(result,status='inconclusive_tangent_refinement')
        tangent=restore_matrix(refinement['refined_derivative_box']);require(tangent.nrows()==size and tangent.ncols()==1,'full refined tangent')
        observe('refined_inputs',{'exact_endpoints':result['exact_endpoints'],'actual_whole_box':vector_json(box),
          'actual_uniform_tangent':vector_json([tangent[i,0] for i in range(size)]),'halfwidth':str((Fraction(right)-Fraction(left))/2)})
        rows=[]
        for j in range(p):
            tick();mixed,prime=total_jacobian_derivative(refined_jet,box[j*n:(j+1)*n],box[-1],[tangent[j*n+i,0] for i in range(n)],tangent[size-1,0])
            row={'phase':j,'center_J':restore_matrix(phases[j]['center_J']),'whole_Jprime':prime};rows.append(row)
            observe('refined_phase',{'phase':j,'center_J':phases[j]['center_J'],'whole_J_mixed_diagnostic':matrix_json(mixed),'whole_total_Jprime':matrix_json(prime)})
        require(frames['cyclic_last_index']==0 and [r['phase'] for r in frames['witnesses']]==list(range(p)),'same complete cyclic frames')
        bases=[restore_matrix(r['B']).transpose() for r in frames['witnesses']]
        witness=framed_cocycle(rows,bases,delta,tick=tick,observe=lambda k,r:observe('refined_'+k,r))
        return dict(result,passed=witness['passed'],status='complete_refined_slice_bounds',norm_cocycle=witness,
          scope='Attraction only on this authenticated primitive-root closed slice if every strict refined gate passes; no neutral-fold or whole-branch statement.')
