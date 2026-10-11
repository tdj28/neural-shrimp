"""Saved implicit second derivative and centered first-derivative enclosure."""
from flint import arb,arb_mat
from .rigorous_connection import require
from .rigorous_orbits import eye,ball_json,matrix_json


def finite_matrix(matrix):
    return all(v.is_finite() for row in matrix.tolist() for v in row)


def implicit_solve(pre,jac,rhs,weights):
    n=jac.nrows();require(jac.ncols()==pre.nrows()==pre.ncols()==rhs.nrows()==len(weights)==n and rhs.ncols()==1,'complete square implicit solve')
    require(all(v.is_finite() and v.is_exact() and v>0 for v in weights),'positive finite exact weights')
    require(all(finite_matrix(v) for v in (pre,jac,rhs)),'finite implicit inputs')
    d=eye(n)-pre*jac
    require(finite_matrix(d),'finite preconditioned defect')
    def norm(v):return max((abs(v[i,0]).upper()/weights[i]).upper() for i in range(n))
    q=max((sum((abs(d[i,j]).upper()*weights[j] for j in range(n)),arb(0))/weights[i]).upper() for i in range(n))
    result={'passed':bool(q.is_finite() and q<1),'weighted_neumann_bound':ball_json(q)}
    if not result['passed']:return result,None
    v=pre*rhs;require(finite_matrix(v),'finite preconditioned RHS')
    radius=(norm(d*v)/(1-q)).upper();require(radius.is_finite() and radius>=0,'finite nonnegative tail radius')
    box=arb_mat([[arb(v[i,0],weights[i]*radius)] for i in range(n)]);rows=[]
    require(finite_matrix(box),'finite initial solution box')
    result.update(preconditioned_rhs=matrix_json(v),initial_box=matrix_json(box),weighted_tail_radius=ball_json(radius))
    for _ in range(3):
        image=v+d*box;require(finite_matrix(image),'finite solution image')
        new=arb_mat([[box[i,0].intersection(image[i,0])] for i in range(n)]);require(finite_matrix(new),'finite solution intersection')
        rows.append({'image':matrix_json(image),'retained':matrix_json(new)});box=new
    result.update(refinements=rows,box=matrix_json(box))
    return result,box


def curvature_forcing(phase_j,phase_jprime,velocity,rate,hidden):
    """A'(s)U'(s): J'_j theta'_j + (partial_h F_j)' h'.

    J' is already the TOTAL phase Jacobian derivative. The extra rate-column
    term supplies the second theta/rate cross term, not a duplicate mistake.
    """
    p=len(phase_j);require(1<=p<=18 and len(phase_jprime)==p,'complete native phases')
    n=phase_j[0].nrows();size=p*n+1
    require(1<=n<=13 and type(hidden) is int and 0<hidden<=n and velocity.nrows()==size and velocity.ncols()==1,'native packing and hidden prefix')
    require(rate.is_finite() and rate>0,'positive actual hidden rate')
    require(all(a.nrows()==a.ncols()==n for a in phase_j+phase_jprime),'native phase matrix sizes')
    require(all(finite_matrix(a) for a in phase_j+phase_jprime+[velocity]),'finite phase and tangent inputs')
    forcing=arb_mat(size,1);rows=[]
    for j,(jac,prime) in enumerate(zip(phase_j,phase_jprime)):
        v=arb_mat([[velocity[j*n+i,0]] for i in range(n)])
        hessian=arb_mat([[((int(i==k)-jac[i,k])/rate) if i<hidden else arb(0) for k in range(n)] for i in range(n)])
        bprime=-hessian*v;value=prime*v+bprime*velocity[size-1,0]
        require(all(finite_matrix(a) for a in (hessian,bprime,value)),'finite native curvature forcing')
        for i in range(n):forcing[j*n+i,0]=value[i,0]
        rows.append({'phase':j,'hidden_hessian_rows':matrix_json(hessian),'parameter_column_derivative':matrix_json(bprime),'forcing':matrix_json(value)})
    return forcing,rows


def refine_tangent(pre,whole_jac,center_jac,phase_j,phase_jprime,velocity,rate,hidden,weights,halfwidth):
    require(halfwidth.is_exact() and halfwidth.is_finite() and halfwidth>0,'fixed positive exact halfwidth')
    require(finite_matrix(velocity),'finite inherited tangent')
    size=velocity.nrows();rhs=arb_mat([[int(i==size-1)] for i in range(size)])
    center,center_box=implicit_solve(pre,center_jac,rhs,weights)
    result={'passed':False,'center_derivative':center}
    if not center['passed']:return result
    forcing,rows=curvature_forcing(phase_j,phase_jprime,velocity,rate,hidden)
    second,second_box=implicit_solve(pre,whole_jac,-forcing,weights)
    result.update(forcing=matrix_json(forcing),native_forcing_rows=rows,second_derivative=second)
    if not second['passed']:return result
    path=center_box+second_box*arb(0,halfwidth)
    require(finite_matrix(path),'finite centered derivative enclosure')
    refined=arb_mat([[velocity[i,0].intersection(path[i,0])] for i in range(size)])
    require(finite_matrix(refined),'finite refined derivative enclosure')
    result.update(passed=True,centered_derivative_enclosure=matrix_json(path),refined_derivative_box=matrix_json(refined),
      theorem='For the same authenticated smooth gauge-root branch, A Uprime=e_last and A Usecond=−Aprime Uprime. The saved total Jprime and −H_hidden theta_prime give every phase of Aprime Uprime, including both actual-rate cross terms. The whole-domain strict Neumann enclosure bounds Usecond uniformly. Integrating from the independently enclosed center Uprime and intersecting the old enclosure retains every true tangent.',
      scope='Conditional derivative enclosure only; no new root, model derivative or attraction claim.')
    return result
