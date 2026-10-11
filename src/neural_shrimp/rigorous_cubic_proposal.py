"""Bounded finite native705 predictor refresh for reusable continuation.

All values are proposals. No root, connection, primitive orbit or stability
claim follows from Newton correction or derivative cancellation alone.
"""
from flint import arb,arb_mat
from .rigorous_unit_pair_pal import assemble,proposal,path_coefficients
from .rigorous_eigenpair_locus import require,finite
from .rigorous_orbits import vector_json,matrix_json,midpoint_matrix


def refresh_cubic_proposal(model,initial,*,period,normal,origin,parameter,newton_steps=2,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;N=3*n*period+3
    require(type(newton_steps)is int and 0<=newton_steps<=2,'fixed finite Newton budget')
    require(all(len(v)==N for v in [initial,normal,origin]),'complete native predictor seed')
    require(all(v.is_exact()and v.is_finite()for v in initial+normal+origin+[parameter]),'finite exact proposal inputs')
    center=list(initial);rhs=arb_mat([[arb(i==N-1)]for i in range(N)])
    for step in range(newton_steps+1):
        # Complete deterministic native input is retained before any phase call;
        # each actual q/r direction return is separately retained by assemble.
        observe('cubic_proposal_input',{'iteration':step,'center':vector_json(center),'normal':vector_json(normal),'origin':vector_json(origin),'parameter':vector_json([parameter])[0],'period':period})
        value,jac,local=assemble(model,center,period,normal,origin,parameter,tick,observe)
        tick();pre,small=proposal(jac,local,n,center[-1]);require(finite(pre),'finite native preconditioner proposal')
        correction=pre*value
        observe('cubic_proposal_return',{'iteration':step,'value':matrix_json(value),'correction':matrix_json(correction),'condensed_border':matrix_json(small),'final_center':step==newton_steps})
        if step<newton_steps:
            require(all(abs(correction[i,0]).upper()<max(arb(1),abs(center[i]).upper())/8 for i in range(N)),'bounded relative proposal correction')
            center=[(center[i]-correction[i,0]).mid()for i in range(N)]
    tangent=midpoint_matrix(pre*rhs);slope=[tangent[i,0]for i in range(N)]
    observe('cubic_affine_proposal_input',{'center':vector_json(center),'slope':vector_json(slope),'normal':vector_json(normal),'origin':vector_json(origin),'parameter':vector_json([parameter])[0]})
    affine=path_coefficients(model,center,slope,[arb(0)]*N,arb(0),period,normal,origin,parameter,tick,observe)
    tick();curve=-pre*arb_mat([[2*affine[i,2]]for i in range(N)]);curvature=[curve[i,0].mid()for i in range(N)]
    observe('cubic_quadratic_proposal_input',{'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json(curvature),'normal':vector_json(normal),'origin':vector_json(origin),'parameter':vector_json([parameter])[0]})
    quadratic=path_coefficients(model,center,slope,curvature,arb(0),period,normal,origin,parameter,tick,observe)
    tick();third=-pre*arb_mat([[quadratic[i,3]]for i in range(N)]);cubic=[third[i,0].mid()for i in range(N)]
    record={'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json(curvature),'cubic':vector_json(cubic),'preconditioner_exact':[[[int(v)for v in x.mid().man_exp()]for x in row]for row in pre.tolist()],'newton_steps':newton_steps,'native_calls':2*period*(newton_steps+3),'algebra_ticks':newton_steps+3,'scope':'Finite predictor/inverse proposals only; a separate fresh full-domain705 proof and complete known-root join are mandatory.'}
    observe('cubic_predictor_return',record)
    return dict(center=center,slope=slope,curvature=curvature,cubic=cubic,C0=pre,record=record)
