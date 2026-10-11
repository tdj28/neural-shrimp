"""Saved shooting blocks -> conditional simple +1 and fold coefficients.

The parent must independently establish a normalized primitive +1 eigenpair
root throughout this domain. This module neither acquires derivatives nor
excludes other unit-circle multipliers. No floating quantity decides a gate.
"""
from flint import arb, arb_mat
from .rigorous_orbits import eye, midpoint_matrix, ball_json, vector_json, matrix_json


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(a):
    return all(a[i,j].is_finite() for i in range(a.nrows()) for j in range(a.ncols()))


def shooting_blocks(jac, box, n, period):
    """Order is (x_j,q_j) per phase, followed by two ACTUAL rates."""
    need(type(n) is int and type(period) is int and 1<=n<=13 and 1<=period<=18, 'dimensions')
    size=2*n*period+2; k=n*period
    need(jac.nrows()==jac.ncols()==len(box)==size, 'full derivative/domain shape')
    need(finite(jac) and all(v.is_finite() for v in box), 'finite whole-domain inputs')
    ix=[2*n*j+i for j in range(period) for i in range(n)]
    iq=[i+n for i in ix]
    a=arb_mat([[jac[i,j] for j in ix] for i in ix])
    q=[box[i] for i in iq]
    r=[arb(0)]*(k-n)+q[:n]
    b=arb_mat(k+1,k+1)
    for i in range(k):
        for j in range(k): b[i,j]=a[j,i]
        b[i,k]=q[i];b[k,i]=r[i]
    rates=arb_mat([[jac[i,size-2+j] for j in range(2)] for i in ix])
    second=[]
    for j in range(period):
        second.extend(sum((jac[2*n*j+n+i,2*n*j+h]*q[j*n+h] for h in range(n)),arb(0)) for i in range(n))
    need(finite(b) and finite(rates) and all(v.is_finite() for v in second), 'finite extracted blocks')
    return b,rates,second


def bounded_solve(b, observe=lambda kind,value:None):
    """Uniform inverse existence and rhs solve, using fixed two weight rules.

    Exact-midpoint C is a proposal; D=I-C*B on the WHOLE matrix and q<1
    certify every B's invertibility. An a-posteriori weighted Neumann tail
    bounds y-y0. No direct interval inverse is assumed to be sufficiently tight.
    """
    k=b.nrows();need(k==b.ncols() and 2<=k<=235 and finite(b), 'finite square border')
    observe('border',{'matrix':matrix_json(b)})
    c=midpoint_matrix(midpoint_matrix(b).inv())
    observe('preconditioner_return',{'matrix':matrix_json(c)})
    need(finite(c), 'finite proposed inverse')
    d=eye(k)-c*b
    observe('defect_return',{'matrix':matrix_json(d)})
    need(finite(d), 'finite whole border defect')
    candidates=[[arb(1)]*k,[sum((abs(c[i,j]).upper() for j in range(k)),arb(0)).upper() for i in range(k)]]
    bounds=[]
    for weights in candidates:
        need(all(v.is_finite() and v>0 for v in weights), 'positive finite fixed weights')
        bounds.append(max((sum((abs(d[i,j]).upper()*weights[j] for j in range(k)),arb(0))/weights[i]).upper() for i in range(k)))
    observe('weight_bounds',{'weights':[vector_json(w) for w in candidates], 'contractions':vector_json(bounds)})
    need(all(v.is_finite() for v in bounds), 'finite contraction bounds')
    index=0 if bounds[0]<=bounds[1] else 1;norm=bounds[index];weights=candidates[index]
    if not norm<1:
        return {'passed':False,'reason':'whole-border contraction inconclusive','contractions':vector_json(bounds),'selected_weight':index},None
    # rhs=e_last; the exact dyadic y0 is C's last column.
    rhs=arb_mat([[int(i==k-1)] for i in range(k)])
    y0=arb_mat([[c[i,k-1]] for i in range(k)])
    residual=rhs-b*y0;forcing=c*residual
    observe('solve_residual_return',{'rhs':matrix_json(rhs),'proposal':matrix_json(y0),'residual':matrix_json(residual),'forcing':matrix_json(forcing)})
    need(finite(residual) and finite(forcing), 'finite solve residual')
    beta=max((abs(forcing[i,0]).upper()/weights[i]).upper() for i in range(k))
    tail=(beta/(1-norm)).upper()
    need(tail.is_finite() and tail>=0, 'finite Neumann tail')
    solution=[y0[i,0]+arb(0,(weights[i]*tail).upper()) for i in range(k)]
    need(all(v.is_finite() for v in solution), 'finite solve enclosure')
    out={'passed':True,'contractions':vector_json(bounds),'selected_weight':index,'beta':ball_json(beta),'tail':ball_json(tail),'solution':vector_json(solution)}
    observe('solution_return',out)
    return out,solution


def qualify(jac,box,n,period,observe=lambda kind,value:None):
    b,rates,second=shooting_blocks(jac,box,n,period)
    observe('coefficient_inputs',{'rates':matrix_json(rates),'second_contractions':vector_json(second)})
    solved,y=bounded_solve(b,observe)
    out={'schema':'neural-shrimp-shooting-adjoint-v1','border_solve':solved,
         'passed':False,'evidence_scope':'Conditional on authenticated whole-domain normalized primitive +1 eigenpair root and derivative enclosure. Periodic-solution fold only; no exclusion of other unit-circle multipliers, attraction, cusp or full geometry theorem.'}
    if y is None:
        out['gates']={'uniform_border_invertibility':False,'simple_native_plus_one':False,'nonzero_rate_transversality':False,'nonzero_quadratic':False}
        return out
    w=y[:-1];alpha=y[-1]
    # alpha=0 is an EXACT-ROOT inference from A Q=0, Q^T Q>0,
    # not an assertion that the independent whole-box coefficients obey it.
    aa=[sum((w[i]*rates[i,j] for i in range(len(w))),arb(0)) for j in range(2)]
    cc=sum((w[i]*second[i] for i in range(len(w))),arb(0))
    need(all(v.is_finite() for v in aa+[cc,alpha]), 'finite coefficients')
    need(alpha.contains(0), 'exact-root alpha=0 compatibility')
    gates={'uniform_border_invertibility':True,'simple_native_plus_one':True,
           'nonzero_rate_transversality':any(not v.contains(0) for v in aa),
           'nonzero_quadratic':not cc.contains(0)}
    out.update(passed=all(gates.values()),gates=gates,left_shooting=vector_json(w),left_native=vector_json(w[-n:]),
               alpha=ball_json(alpha),actual_rate_coefficients=vector_json(aa),second_derivative_contraction=ball_json(cc),quadratic_coefficient=ball_json(cc/2),
               normalization='w_(period-1)^T q0=1; q0^T q0=1; c=w_native^T Dxx(F^period)[q0,q0], reduced quadratic coefficient c/2')
    observe('qualification_return',out)
    return out
