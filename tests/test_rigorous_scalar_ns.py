import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_scalar_ns import (Jet,augmented,inclusion,coefficient,derivative_data,
    local_conditions,scalar_step,scalar_jets,check_scalar_ns)
from neural_shrimp.rigorous_orbits import exact


def normal_form(z,kappa=-1,quadratic=False,tangent=False,resonant=False):
    u,v,a=[Jet.variable(z[i],i) for i in range(3)]
    c,s=(arb(0),arb(1)) if resonant else (exact('3/5'),exact('4/5'))
    scale=1+(a*a if tangent else a)+kappa*(u*u+v*v)
    out=[scale*(c*u-s*v),scale*(s*u+c*v)]
    if quadratic:
        out=[out[0]+exact('1/2')*u*u+exact('1/3')*u*v,
             out[1]-exact('2/3')*v*v+exact('1/5')*u*v]
    return out


def test_jet_factorials_and_mixed_derivatives_exact():
    with ctx.workprec(256):
        x,y,z=[Jet.variable(arb(0),i) for i in range(3)]
        f=7*x*x*x+5*x*y*y+3*x*y*z+2*z
        assert f.derivative(0,0,0)==42
        assert f.derivative(0,1,1)==10
        assert f.derivative(0,1,2)==3
        assert f.derivative(2)==2
        assert f.derivative(0,0)==0


def test_scalar_polynomial_jacobian_matches_independent_composition():
    with ctx.workprec(256):
        p=[exact('1/4'),exact('-3/4'),exact('1/8')]
        target,ridge,ratio=exact('5/4'),exact('1/4'),exact('2')
        step=lambda x:scalar_step(x,p[2],target,ridge,ratio)
        y=step(p[:2]);end=step(y)
        def J(x):
            u,v=x;eta=p[2]
            return [[1-eta*(v*v+ridge),-eta*(2*u*v-target)],
                    [-ratio*eta*(2*u*v-target),1-ratio*eta*(u*u+ridge)]]
        J0,J1=J(p[:2]),J(y)
        jets=scalar_jets(p,target,ridge,ratio)
        for i in range(2):
            assert jets[i].derivative().contains(end[i])
            for j in range(2):
                expected=sum((J1[i][k]*J0[k][j] for k in range(2)),arb(0))
                assert (jets[i].derivative(j)-expected).contains(0)
        # Parameter derivative checked by exact rational central difference with an O(h^2) tolerance.
        h=arb((1,-40));zp=p.copy();zm=p.copy();zp[2]+=h;zm[2]-=h
        fp=scalar_step(scalar_step(zp[:2],zp[2],target,ridge,ratio),zp[2],target,ridge,ratio)
        fm=scalar_step(scalar_step(zm[:2],zm[2],target,ridge,ratio),zm[2],target,ridge,ratio)
        assert all(abs((fp[i]-fm[i])/(2*h)-jets[i].derivative(2))<arb((1,-70)) for i in range(2))


@pytest.mark.parametrize('kappa',[-1,1,0])
def test_normal_form_coefficient_has_exact_known_sign_and_scale(kappa):
    with ctx.workprec(256):
        jets=normal_form([arb(0)]*3,kappa)
        A,B,C,_,_=derivative_data(jets)
        a,b,details=coefficient(A,B,C)
        assert a.contains(2*kappa) and b.contains(2*kappa)
        assert abs(a-b)<arb((1,-200))
        assert details['both_strictly_negative']==(kappa<0)
        local=local_conditions(jets)
        assert local['positive_transversality'] and local['nonresonance_orders_1_to_4']


def test_independent_planar_and_resolvent_formulas_agree_with_quadratic_terms():
    with ctx.workprec(256):
        for kappa in [-1,1]:
            A,B,C,_,_=derivative_data(normal_form([arb(0)]*3,kappa,quadratic=True))
            a,b,_=coefficient(A,B,C)
            assert (a-b).contains(0) and abs(a-b)<arb((1,-190))


def test_augmented_inclusion_and_tangent_resonance_fail_closed():
    with ctx.workprec(256):
        center=[arb(0)]*3
        box,result=inclusion(center,arb((1,-40)),normal_form)
        assert result['passed']
        with pytest.raises((ZeroDivisionError,ValueError)):
            inclusion(center,arb((1,-40)),lambda z:normal_form(z,tangent=True))
        local=local_conditions(normal_form(center,resonant=True))
        assert not local['nonresonance_orders_1_to_4'] and not local['negative_criticality']
        H,_=augmented(center,normal_form)
        assert all(x.contains(0) for x in H)


def test_collapsed_state_and_invalid_exact_input_are_not_proofs():
    with ctx.workprec(256):
        out=scalar_step([arb(0),arb(0)],exact('1/4'),exact('5/4'),exact('1/4'),exact('2'))
        assert all(x==0 for x in out)
        bad=dict(schema='scalar-ns-input-v1',target='6/5',ridge='1/4',ratio='4',approximation=['0','0','1'])
        with pytest.raises(ValueError,match='dyadic'):check_scalar_ns(bad)
        with pytest.raises(ValueError,match='recipe'):check_scalar_ns(bad,precision=128)


def test_wide_inclusion_and_zero_straddling_sign_fail_closed():
    with ctx.workprec(256):
        _,result=inclusion([arb(0)]*3,arb(1),normal_form)
        assert not result['passed']
        A,B,C,_,_=derivative_data(normal_form([arb(0)]*3,arb(0,1)))
        ell,other,details=coefficient(A,B,C)
        assert ell.contains(0) and other.contains(0) and not details['both_strictly_negative']


def test_offline_checker_rejects_bad_source_inventory_and_seed(tmp_path):
    import importlib.util,json
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'tools/certify_scalar_ns.py'
    spec=importlib.util.spec_from_file_location('scalar_ns_driver_test',script)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    seed=tmp_path/'seed.json';seed.write_text('{}')
    with pytest.raises(ValueError,match='seed receipt hash'):driver.inputs_from_seed(seed)
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(schema='scalar-ns-receipt-v1',files={k:{} for k in driver.FILES},source_sha256={})))
    with pytest.raises(ValueError,match='source inventory'):driver.verify(receipt)
    receipt.write_text(json.dumps(dict(schema='scalar-ns-receipt-v1',files={k:{} for k in driver.FILES},source_sha256={k:'corrupt' for k in driver.SOURCES})))
    with pytest.raises(ValueError,match='source hash'):driver.verify(receipt)
