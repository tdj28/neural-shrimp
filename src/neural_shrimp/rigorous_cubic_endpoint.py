"""Conditional localization of a known cubic-tube endpoint; no model calls."""
from flint import arb, arb_mat
from .rigorous_connection import require
from .rigorous_parameter_localization import finite, fixed_point_box
from .rigorous_orbits import ball_json, matrix_json, vector_json


def localize_cubic_endpoint(defect, forcing, d, halfwidth, predictor, known, weights, error, *, tick=lambda:None):
    """Retain a known root with e=v(d)+D_actual(d,e)e at one exact d.

    Caller binds the coefficients to the SAME already proved cubic tube:
    D(d)=sum(D_k d^k) encloses I-C(d)H_U on the full correction box;
    v(d)=sum(v_k d^k) encloses -C(d)H(p(d),s_c+d).
    The old root must be the endpoint at this exact scalar, with original
    correction-domain membership inherited from its existence theorem.
    Coefficient intervals need not be independent. There are no new native
    evaluations, inverse proposals, roots or parameter extensions here.
    """
    n=len(weights)
    require(1<=n<=705 and len(defect)==4 and len(forcing)==6,'full cubic polynomial dimensions')
    require(all(len(v)==n for v in (predictor,known,error)),'all known endpoint coordinates')
    require(d.is_exact()and halfwidth.is_exact()and halfwidth>0 and abs(d)<=halfwidth,'exact point inside proved parameter interval')
    require(all(v.is_finite()and v.is_exact()for v in predictor),'exact finite endpoint predictor')
    require(all(v.is_finite()and v.is_exact()and v>0 for v in weights),'positive exact weights')
    require(all(v.is_finite()for v in known+error)and all(v.contains(0)for v in error),'finite original correction and known box')
    require(all(v.nrows()==v.ncols()==n and finite(v)for v in defect)and all(v.nrows()==n and v.ncols()==1 and finite(v)for v in forcing),'full finite saved polynomial arrays')
    D=sum((v*d**k for k,v in enumerate(defect)),arb_mat(n,n))
    rhs=sum((v*d**k for k,v in enumerate(forcing)),arb_mat(n,1));tick()
    rows=[(sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)]
    kappa=max(rows);require(kappa.is_finite()and kappa<1,'strict pointwise weighted Neumann contraction')
    initial=arb_mat([[(known[i]-predictor[i]).intersection(error[i])]for i in range(n)])
    correction,proof=fixed_point_box(D,rhs,weights,kappa,initial=initial,iterations=3);tick()
    box=[known[i].intersection(predictor[i]+correction[i,0])for i in range(n)]
    return box,dict(passed=True,dimension=n,parameter_offset=ball_json(d),halfwidth=ball_json(halfwidth),predictor=vector_json(predictor),weighted_neumann_bound=ball_json(kappa),weighted_row_bounds=vector_json(rows),preconditioned_rhs=matrix_json(rhs),refinement=proof,box=vector_json(box),scope='Conditional enclosure of the identical known endpoint using authenticated full-error polynomial derivatives and residual. No new root, native acquisition, parameter extension, or complementary stability.')
