import pytest
from fractions import Fraction
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.eigenpair_width_diagnostic import split_width
from neural_shrimp.rigorous_eigenpair_march import decode
from test_eigenpair_reanchor import parent


def test_exact_polynomial_split_degree_scaling_and_no_wider_proof():
    with ctx.workprec(256):
        _,c=parent();n=c['shooting_dimension'];b=lambda x:ball_json(arb(x));z=b(0)
        c['preconditioner']=[[b(i==j) for j in range(n)] for i in range(n)]
        c['center_jacobian']=[[b(1 if i==j else 0) for j in range(n)] for i in range(n)]
        c['defect']=[[b('.1') if i==j else z for j in range(n)] for i in range(n)]
        c['coordinate_radii']=[b(1)]*n;c['settings']['halfwidth_exponent']=-2
        c['quadratic_center_coefficients']=[[b(1),b(2),b(3),b(999)] for i in range(n)]
        c['quadratic_whole_coefficients']=[[z,z,z,b(4)] for i in range(n)]
        events=[];r=split_width(c,lambda k,v:events.append(k));s=r['summary']
        assert decode(s['forcing_degree_max'][3]).contains(arb(4)/64)
        for key,truth in (('same_width_triangle_proxy_max',Fraction(37,20)),('double_width_proxy_max',Fraction(69,20))):
            v=s[key];assert v['rad'][0]==0
            bound=Fraction(v['mid'][0])*Fraction(2)**v['mid'][1]
            assert truth<=bound<truth+Fraction(2)**-200
        assert s['double_width_proxy_below_one'] is False and 'no proof gate' in r['scope']
        assert events==['center_defect_return']+['coefficient_action_return']*4+['width_budget_return']


def test_identity_zero_forcing_proxy_and_unproved_rejection():
    with ctx.workprec(256):
        _,c=parent();n=c['shooting_dimension'];b=lambda x:ball_json(arb(x));z=b(0)
        c['preconditioner']=[[b(i==j) for j in range(n)] for i in range(n)];c['center_jacobian']=c['preconditioner'];c['defect']=[[z]*n for i in range(n)]
        c['quadratic_center_coefficients']=[[z]*4 for i in range(n)];c['quadratic_whole_coefficients']=[[z]*4 for i in range(n)]
        assert split_width(c)['summary']['double_width_proxy_below_one'] is True
        c['passed']=False
        with pytest.raises(ValueError):split_width(c)


def test_driver_complete_sources_and_no_native_acquisition():
    import importlib.util
    from pathlib import Path
    root=Path(__file__).parents[1];p=root/'tools/diagnose_eigenpair_width.py';spec=importlib.util.spec_from_file_location('width_driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert set(m.T.SOURCES)<=set(m.SOURCES) and all((root/n).is_file() for n in m.SOURCES)
    text=p.read_text();assert 'TwoRateCuspMap(' not in text and 'assemble(' not in text and 'limit=45' in text and 'rss=1536*1024' in text
    assert '19/128' in text and m.PARENT_SOURCE=='5f32fccb71262f3106fcb2e69dce0f5c5dcfe033'
