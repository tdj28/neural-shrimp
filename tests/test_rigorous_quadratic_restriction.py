import pytest
from flint import arb, arb_mat, ctx
from neural_shrimp.rigorous_quadratic_restriction import restrict_quadratic
from neural_shrimp.rigorous_quadratic_localization import localize_quadratic_root
from neural_shrimp.rigorous_connection import restore_ball


def data(shift):
    h=arb('1/16');delta=arb(0,h)
    return dict(center=[arb(1)],slope=[arb('1/2')],curvature=[arb('-1/4')],
                weights=[arb(1)],original_halfwidth=h,radius=arb('1/1024'),
                shift=arb(shift),halfwidth=h/2,
                center_coefficients=arb_mat([[0,0,0,arb('-1/8')]]),
                whole_coefficients=arb_mat([[0,0,0,arb('-1/8')+delta/16]]))


@pytest.mark.parametrize('shift',['-1/32','1/32'])
def test_exact_polynomial_identity_and_inherited_root_correction(shift):
    with ctx.workprec(256):
        a=data(shift);r=restrict_quadratic(**a)
        for j in range(-16,17):
            d=arb(j)/512;t=a['shift']+d
            old=1+t/2-t*t/8;new=r['center'][0]+r['slope'][0]*d+r['curvature'][0]*d*d/2
            assert (old-new).is_zero()
            root=(1+t).sqrt()
            assert abs(root-new).upper()<a['radius']
            assert r['fresh_domain'][0].contains(root)
            assert r['path_residual'][0,0].contains(old*old-(1+t))
            assert r['residual_derivative'][0,0].contains(-3*t*t/8+t*t*t/16)


@pytest.mark.parametrize('shift',['-1/32','1/32'])
def test_fresh_jacobian_localizes_known_nonlinear_root_without_newton(shift):
    with ctx.workprec(256):
        a=data(shift);r=restrict_quadratic(**a);c=r['center'][0]
        out=localize_quadratic_root(C=arb_mat([[(1/(2*c)).mid()]]),
            J=arb_mat([[2*r['fresh_domain'][0]]]),
            G0=arb_mat([[c*c-(1+a['shift'])]]),
            parameter_partial=arb_mat([[-1]]),
            **{k:r[k]for k in ['center','slope','curvature','weights','halfwidth','radius','path_residual']})
        whole=restore_ball(out['whole_box'][0]);tangent=restore_ball(out['tangent_box'][0][0])
        for j in range(-16,17):
            t=a['shift']+arb(j)/512;root=(1+t).sqrt()
            assert whole.contains(root) and tangent.contains(1/(2*root))


def test_parameter_escape_and_inexact_recenter_inputs_rejected():
    with ctx.workprec(256):
        a=data('1/32');a['shift']+=arb('1/1024')
        with pytest.raises(ValueError,match='within proved'):restrict_quadratic(**a)
        a=data('1/32');a['center'][0]=arb(1,arb('1e-40'))
        with pytest.raises(ValueError,match='exact finite'):restrict_quadratic(**a)
        a=data('1/32');a['whole_coefficients']=arb_mat(1,3)
        with pytest.raises(ValueError,match='Taylor'):restrict_quadratic(**a)
