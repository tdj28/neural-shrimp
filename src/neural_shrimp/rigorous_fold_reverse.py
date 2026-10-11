"""Proposal-only fold-native tangent and coordinate scales for unchanged charts."""
from flint import arb,arb_mat
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import fraction_mid
from .rigorous_orbits import exact,vector_json,matrix_json,ball_json,midpoint_matrix
from .rigorous_arclength_connection import bordered


def fold_native_proposal(model,orbit,rate,right,target_box,*,tick=lambda:None,observe=lambda k,r:None):
    """18 native derivative proposals plus one bordered point acquisition.

    Frozen dyadic normal/weights have no theorem status. Actual chart gates
    reacquire their own entire-domain derivatives and identify the whole fold
    box at its correlated gauge. Target informs orientation only.
    """
    period=len(orbit);n=model.n;size=period*n+1
    require(1<=period<=18 and all(len(row)==n for row in orbit),'native orbit dimensions')
    require(len(right)==n and len(target_box)==size,'full fold and target dimensions')
    box=[restore_ball(v) for row in orbit for v in row]+[restore_ball(rate)]
    center=[v.mid() for v in box];q=[restore_ball(v).mid() for v in right]
    require(all(v.is_finite() for v in center+q) and center[-1]>0,'finite positive-rate proposal')
    require(any(v!=0 for v in q),'nonzero initial fold null proposal')
    first=q.copy();tangent=[];steps=[]
    for j in range(period):
        tick();image,jac,partial=model.step(center[j*n:(j+1)*n],center[-1])
        tangent.extend(q);nextq=jac*arb_mat([[v] for v in q])
        steps.append({'phase':j,'point_state':vector_json(center[j*n:(j+1)*n]),'jacobian':matrix_json(jac),
          'image':vector_json(image),'partial':vector_json(partial),'direction_before':vector_json(q),'direction_after':matrix_json(nextq)})
        observe('fold_tangent_phase',steps[-1])
        q=[nextq[i,0].mid() for i in range(n)]
    tangent.append(arb(0))
    direction=[restore_ball(v).mid()-center[i] for i,v in enumerate(target_box)]
    dot=sum((tangent[i]*direction[i] for i in range(size)),arb(0))
    require(dot.is_finite() and not dot.contains(0),'unambiguous target-informed orientation')
    sign=1 if dot>0 else -1
    tangent=[sign*v for v in tangent];length=sum((v*v for v in tangent),arb(0)).sqrt()
    require(length.is_finite() and length>0,'finite tangent norm')
    normal=[exact(str(fraction_mid(v/length))) for v in tangent]
    require(normal[-1]==0 and any(v!=0 for v in normal),'state-only fold normal')
    residual,jac,blocks=bordered(model,center,period,normal,center,arb(0),tick)
    observe('fold_point_border',{'normal':vector_json(normal),'center':vector_json(center),'residual':matrix_json(residual),'jacobian':matrix_json(jac)})
    tick();pre=midpoint_matrix(jac.inv())
    row_sizes=[sum((abs(pre[i,j]) for j in range(size)),arb(0)) for i in range(size)]
    largest=max(v.upper() for v in row_sizes);require(largest.is_finite() and largest>0,'finite preconditioner scale')
    floor=arb((1,-12));weights=[exact(str(fraction_mid(max(floor,v/largest)))) for v in row_sizes]
    require(all(v.is_exact() and v.is_finite() and 0<v<=1 for v in weights),'positive dyadic row scales')
    return {'normal':[str(fraction_mid(v)) for v in normal],'origin':[str(fraction_mid(v)) for v in center],
      'initial':[str(fraction_mid(v)) for v in center],'weights':[str(fraction_mid(v)) for v in weights],
      'anchor':{'state_box':vector_json(box)},'period':period,'derivative_proposals':steps,
      'propagated_closure_defect':vector_json([q[i]-first[i] for i in range(n)]),
      'orientation_dot_before_sign':ball_json(dot),'orientation_sign':sign,
      'point_bordered_residual':matrix_json(residual),'point_bordered_jacobian':matrix_json(jac),
      'point_preconditioner':matrix_json(pre),'point_row_sizes':vector_json(row_sizes),
      'scope':'target-informed proposal only; no exact nullity, connection or whole-domain invertibility asserted'}
