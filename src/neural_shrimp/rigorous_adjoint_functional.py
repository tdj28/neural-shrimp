"""Residual bounds of scalar functions of a known interval linear solution.

For By=rhs and any fixed exact u, b^T y=u^T rhs+(b-B^T u)^T y.
B,b may depend on the same root; independent interval evaluation stays valid.
No solve, inverse or new derivative is needed. Supplied Y/B provenance matters.
"""
from flint import arb,arb_mat
from .rigorous_shooting_adjoint import need,finite
from .rigorous_orbits import midpoint_matrix,matrix_json,vector_json,ball_json


def functional(border,pre,solution,b,observe=lambda kind,value:None):
    k=len(solution)
    need(2<=k<=235 and border.nrows()==border.ncols()==pre.nrows()==pre.ncols()==k and len(b)==k,'functional shape')
    need(finite(border) and finite(pre) and all(v.is_finite() for v in solution+b),'finite functional inputs')
    need(all(pre[i,j].is_exact() for i in range(k) for j in range(k)),'fixed exact proposal preconditioner')
    need(b[-1].is_exact() and b[-1]==0 and solution[-1].is_exact() and solution[-1]==0,'root-only auxiliary zero')
    u=midpoint_matrix(pre.transpose()*arb_mat([[v.mid()] for v in b]))
    observe('dual_proposal_return',{'b':vector_json(b),'proposal':matrix_json(u)})
    need(finite(u) and all(u[i,0].is_exact() for i in range(k)),'finite exact dual proposal')
    residual=arb_mat([[v] for v in b])-border.transpose()*u
    observe('dual_residual_return',{'residual':matrix_json(residual)})
    need(finite(residual),'finite dual residual')
    direct=sum((b[i]*solution[i] for i in range(k)),arb(0))
    dual=u[k-1,0]+sum((residual[i,0]*solution[i] for i in range(k)),arb(0))
    observe('functional_bounds_return',{'direct':ball_json(direct),'dual':ball_json(dual)})
    need(direct.is_finite() and dual.is_finite(),'finite scalar bounds')
    intersection=direct.intersection(dual);need(intersection.is_finite(),'nonempty scalar intersection')
    out={'direct':ball_json(direct),'dual':ball_json(dual),'intersection':ball_json(intersection),'excludes_zero':not intersection.contains(0),
         'scope':'Conditional scalar coefficient enclosure from the authenticated exact-root linear equation; no extra dynamics or stability assumption.'}
    observe('functional_intersection_return',out)
    return out,intersection
