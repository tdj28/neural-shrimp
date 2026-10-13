"""High-precision multiple-shooting midpoint proposals, never proof gates.

Fixed exact dyadic A_i and B approximate derivatives and (I-A_{p-1}...A_0)^-1.
The enclosure F(c_i) supplied here must have been computed separately. This
module keeps residual/correction balls, then deliberately chooses exact midpoint
centres for a later evaluation. That midpoint choice is NOT an inclusion proof.
"""
from flint import arb, arb_mat
from proof_gates import ball_record, matrix_record, vector_record


def _vector(values, n=None, *, point=False):
    if not isinstance(values, (tuple, list)) or not values or n is not None and len(values) != n:
        raise ValueError('nonempty consistent state vectors required')
    if any(not isinstance(v, arb) or not v.is_finite() or point and not v.is_exact() for v in values):
        raise ValueError('finite Arb vectors required; centres must be exact dyadic points')
    return values


def _matrix(matrix, n):
    if not isinstance(matrix, arb_mat) or (matrix.nrows(), matrix.ncols()) != (n,n):
        raise ValueError('full square preconditioner proposals required')
    if any(not matrix[i,j].is_finite() or not matrix[i,j].is_exact() for i in range(n) for j in range(n)):
        raise ValueError('A and B must be finite exact dyadic proposals')
    return matrix


def correct(centers, images, derivatives, inverse_cycle):
    """Return next exact centres plus complete residual/correction witnesses.

    h_i=F(c_i)-c_(i+1). For delta=-R h, recurrence is
    s_(i+1)=A_i s_i+h_i, s_0=0; delta_0=B s_p;
    delta_(i+1)=A_i delta_i+h_i. This sign convention solves
    A_i delta_i-delta_(i+1)=-h_i when B is the exact cyclic inverse.
    B need only be a fixed proposal here; its inverse defect is not asserted.
    """
    if not isinstance(centers,(tuple,list)) or len(centers)<2:
        raise ValueError('at least two phases required')
    p=len(centers); n=len(_vector(centers[0],point=True))
    if not isinstance(images,(tuple,list)) or not isinstance(derivatives,(tuple,list)) or len(images)!=p or len(derivatives)!=p:
        raise ValueError('one image and derivative proposal per phase required')
    for v in centers:_vector(v,n,point=True)
    for v in images:_vector(v,n)
    for A in derivatives:_matrix(A,n)
    B=_matrix(inverse_cycle,n)
    h=[arb_mat(n,1,[images[i][j]-centers[(i+1)%p][j] for j in range(n)]) for i in range(p)]
    s=arb_mat(n,1)
    for A,residual in zip(derivatives,h):s=A*s+residual
    delta=B*s; first=delta; corrections=[]; next_centers=[]
    for i,A in enumerate(derivatives):
        if any(not delta[j,0].is_finite() for j in range(n)):raise ValueError('nonfinite correction')
        corrections.append([delta[j,0] for j in range(n)])
        next_centers.append([(centers[i][j]+delta[j,0]).mid() for j in range(n)])
        delta=A*delta+h[i]
    closure=delta-first
    if any(not closure[j,0].is_finite() for j in range(n)):raise ValueError('nonfinite cyclic closure')
    upper=max(h[i][j,0].abs_upper() for i in range(p) for j in range(n))
    return dict(centers=next_centers,
                record=dict(schema='codex-shooting-midpoint-proposal-v1',certificate_passed=False,
                    original_centers=[vector_record(v) for v in centers],
                    images=[vector_record(v) for v in images],
                    residuals=[matrix_record(v) for v in h],
                    residual_inf_upper=ball_record(upper),
                    corrections=[vector_record(v) for v in corrections],
                    linear_cyclic_defect=matrix_record(closure),
                    next_centers=[vector_record(v) for v in next_centers],
                    note='Radii retained in witnesses, deliberately discarded only when selecting new proposal midpoints; recomputation and all proof gates still required.'))
