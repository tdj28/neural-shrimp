import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_cubic_restriction import restrict_cubic

def fixture(shift=arb(1)/8,halfwidth=arb(1)/16):
    c=[arb(1),arb(2)];s=[arb(2),arb(3)];a=[arb(3),arb(4)];b=[arb(4),arb(5)];w=[arb(1),arb(2)];C=arb_mat([[1,2,3,4,5],[2,3,4,5,6]]);return c,s,a,b,w,arb(1)/4,arb((1,-10)),shift,halfwidth,C,C

@pytest.mark.parametrize("sign",[-1,0,1])
def test_exact_cubic_translation_all_coefficients_and_inherited_root(sign):
    with ctx.workprec(256):
        args=fixture(arb(sign)/8);r=restrict_cubic(*args);c,s,a,b,w,H,rho,shift,h,_,_=args
        for d in [arb(-1)/16,arb(0),arb(1)/16]:
            t=shift+d
            for i in range(2):
                old=c[i]+s[i]*t+a[i]*t*t/2+b[i]*t*t*t
                new=r['center'][i]+r['slope'][i]*d+r['curvature'][i]*d*d/2+r['cubic'][i]*d*d*d
                assert old==new and r['fresh_domain'][i].contains(old+arb(0,rho*w[i]))
                assert r['predictor_derivative'][i].contains(s[i]+a[i]*t+3*b[i]*t*t)

def test_separate_residual_and_derivative_taylor_theorems():
    with ctx.workprec(256):
        args=list(fixture());args[9]=arb_mat([[0]*5,[0]*5]);args[10]=arb_mat([[0,0,0,0,arb(0,arb(5)/4)],[0,0,0,0,1]])
        # R1=t^5: R1''''(I)/4!=5I. R2=t^4: coefficient=1.
        r=restrict_cubic(*args)
        for t in [arb(1)/16,arb(1)/8,arb(3)/16]:
            assert r['path_residual'][0,0].contains(t**5)and r['residual_derivative'][0,0].contains(5*t**4)
            assert r['path_residual'][1,0].contains(t**4)and r['residual_derivative'][1,0].contains(4*t**3)

def test_subset_and_exact_coefficient_guards():
    with ctx.workprec(256):
        with pytest.raises(ValueError):restrict_cubic(*fixture(arb(1)/4,arb(1)/16))
        args=list(fixture());args[3]=[arb(1,arb((1,-80))),arb(2)]
        with pytest.raises(ValueError):restrict_cubic(*args)
        args=list(fixture());args[10]=arb_mat(2,4)
        with pytest.raises(ValueError):restrict_cubic(*args)
