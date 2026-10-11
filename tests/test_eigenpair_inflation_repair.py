from fractions import Fraction
import copy
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_inflation_repair import outward_dyadic,repair_inputs,repair
from neural_shrimp.rigorous_orbits import exact


def proposal(c):
    old=[Fraction(v)*Fraction(2)**c['settings']['radius_exponent'] for v in c['weights']]
    return {'old_exact_radii':list(map(str,old)),'exact_radii':[str(v*(1+Fraction(1,1000)+Fraction(2)**-300)) for v in old]}


def test_outward_rounding_precise_and_representable():
    with ctx.workprec(256):
        for exp in [-90,-20,0,12]:
            v=Fraction(2)**exp*(1+Fraction(2)**-129+Fraction(2)**-301)
            rounded=outward_dyadic(v)
            assert rounded>=v and rounded/v<1+Fraction(2)**-126
            assert exact(str(rounded)).is_exact()
        assert outward_dyadic(Fraction(1,8))==Fraction(1,8)


@pytest.mark.parametrize('sigma',[-1,1])
def test_one_fresh_chart_uses_last_qualified_parent_not_failed_tail(sigma):
    with ctx.workprec(256):
        model,c=parent(sigma);p=proposal(c);initial,kw,audit=repair_inputs(c,p)
        assert kw['normal']==c['normal'] and kw['origin']==c['origin']
        assert kw['anchor']['exact_parameter']==c['parameter_endpoints'][1]
        assert all(exact(v).is_exact() for v in kw['weights'])
        assert all(Fraction(a)>=Fraction(b) for a,b in zip(audit['rounded_exact_radii'],p['exact_radii']))
        ticks=[];events=[];r=repair(model,c,p,tick=lambda:ticks.append(1),observe=lambda k,v:events.append(k))
        assert r['passed'] and len(ticks)==15 and events.count('chart_return')==1
        assert r['parameter_endpoints'][0]==c['parameter_endpoints'][1]
        assert all(r['gates'].values()) and r['sigma']==sigma


def test_radius_join_and_unproved_parent_fail_before_calls():
    with ctx.workprec(256):
        _,c=parent();p=proposal(c);p['old_exact_radii'][0]='1'
        with pytest.raises(ValueError):repair_inputs(c,p)
        p=proposal(c);p['exact_radii'][0]='1/1000000000000'
        with pytest.raises(ValueError):repair_inputs(c,p)
        p=proposal(c);c['passed']=False
        with pytest.raises(ValueError):repair_inputs(c,p)


def test_failed_proposal_durability_prevents_all_model_calls():
    with ctx.workprec(256):
        model,c=parent();ticks=[]
        def observer(k,v):raise OSError('save failure')
        with pytest.raises(OSError):repair(model,c,proposal(c),tick=lambda:ticks.append(1),observe=observer)
        assert ticks==[]
