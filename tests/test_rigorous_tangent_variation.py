import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_tangent_variation import curvature_forcing,implicit_solve,refine_tangent
from neural_shrimp.rigorous_orbits import eye
from neural_shrimp.rigorous_connection_frames import restore_matrix


def test_two_rate_cross_terms_and_output_packing():
    # F0=x0-h*x0^2 and F1=x1-3*x1^2, at x=(2,4), h=5,
    # xprime=(7,11), hprime=13: Jprime0=-2h*xprime0-2x0*hprime.
    j=arb_mat([[-19,0],[0,-23]]);jp=arb_mat([[-122,0],[0,-66]])
    f,rows=curvature_forcing([j],[jp],arb_mat([[7],[11],[13]]),arb(5),1)
    assert f==arb_mat([[-1218],[-726],[0]])
    assert restore_matrix(rows[0]['parameter_column_derivative'])==arb_mat([[-28],[0]])


def test_implicit_solve_encloses_exact_solution_and_gauge():
    with ctx.workprec(256):
        a=arb_mat([[2,1],[0,1]]);pre=arb_mat([['1/2','-1/2'],[0,1]])
        proof,box=implicit_solve(pre,a,arb_mat([[3],[1]]),[arb(1),arb(1)])
        assert proof['passed'] and box[0,0].contains(1) and box[1,0].contains(1)
        failed,_=implicit_solve(eye(2),arb_mat([[3,0],[0,3]]),arb_mat([[0],[1]]),[arb(1)]*2)
        assert not failed['passed']


def test_centered_tangent_exact_parabolic_branch():
    # H(theta,h)=h-theta^2, gauge theta=s. At s=2, U'=(1,4), U''=(0,2).
    # A=[[-2theta,1],[1,0]], J=1-2theta, Jprime=-2.
    # Use hidden=n and h=1 artificially with F_h=1 constant: the GD Hessian
    # relation would not represent this model, so test the general solve only.
    with ctx.workprec(256):
        a=arb_mat([[-4,1],[1,0]]);pre=arb_mat([[0,1],[1,4]])
        p,b=implicit_solve(pre,a,arb_mat([[2],[0]]),[arb(1)]*2)
        assert p['passed'] and b==arb_mat([[0],[2]])


def test_constant_linear_gd_root_derivative_exact():
    # theta=0 on F=(1-h)theta, h=s; U'=(0,1), U''=0.
    with ctx.workprec(256):
        a=arb_mat([[-arb('1/4'),0],[0,1]]);pre=arb_mat([[-4,0],[0,1]])
        result=refine_tangent(pre,a,a,[arb_mat([['3/4']])],[arb_mat([[-1]])],arb_mat([[0],[1]]),arb('1/4'),1,[arb(1)]*2,arb('1/32'))
        assert result['passed'] and restore_matrix(result['refined_derivative_box'])==arb_mat([[0],[1]])


@pytest.mark.parametrize('rate',[arb(0),arb(-1),arb(0,1)])
def test_nonpositive_actual_rate_rejected(rate):
    with pytest.raises(ValueError):curvature_forcing([eye(1)],[eye(1)],arb_mat([[1],[1]]),rate,1)


def test_missing_native_phase_is_rejected():
    with pytest.raises(ValueError):curvature_forcing([eye(1)],[],arb_mat([[1],[1]]),arb(1),1)


@pytest.mark.parametrize('which',['pre','jac','rhs','weight'])
def test_nonfinite_implicit_input_never_passes(which):
    args={'pre':eye(1),'jac':eye(1),'rhs':arb_mat([[1]]),'weights':[arb(1)]}
    if which=='weight':args['weights']=[arb('inf')]
    else:args[which]=arb_mat([[arb('nan')]])
    with pytest.raises(ValueError,match='finite'):implicit_solve(**args)


@pytest.mark.parametrize('which',['jac','prime','velocity'])
def test_nonfinite_curvature_input_never_passes(which):
    j=eye(1);jp=eye(1);v=arb_mat([[1],[1]])
    if which=='jac':j[0,0]=arb('nan')
    elif which=='prime':jp[0,0]=arb('nan')
    else:v[0,0]=arb('nan')
    with pytest.raises(ValueError,match='finite'):curvature_forcing([j],[jp],v,arb(1),1)
