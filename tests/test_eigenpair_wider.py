from fractions import Fraction
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_wider import wider_inputs,widen
from neural_shrimp.rigorous_eigenpair_march import decode


@pytest.mark.parametrize('sigma',[-1,1])
def test_doubled_width_starts_exactly_at_parent_and_reacquires_native_proof(sigma):
    with ctx.workprec(256):
        m,c=parent(sigma);initial,kw=wider_inputs(c);old=Fraction(2)**c['settings']['halfwidth_exponent']
        assert Fraction(kw['parameter'])-2*old==Fraction(c['parameter_endpoints'][1])
        assert kw['weights']==c['weights'] and kw['normal']==c['normal'] and kw['origin']==c['origin']
        assert all(a.contains(decode(b)) for a,b in zip(kw['anchor']['box'],c['endpoint_root_boxes'][1]))
        ticks=[];events=[];r=widen(m,c,tick=lambda:ticks.append(1),observe=lambda k,v:events.append(k))
        assert r['passed'] and len(ticks)==30 and events.count('path_phase_return')==12 and events.count('phase_return')==16
        assert r['charts'][0]['parameter_endpoints'][0]==c['parameter_endpoints'][1]
        assert r['charts'][0]['parameter_endpoints'][1]==r['charts'][1]['parameter_endpoints'][0]
        assert all(v['sigma']==sigma and all(v['gates'].values()) for v in r['charts'])


def test_unqualified_or_inconsistent_parent_cannot_launch():
    with ctx.workprec(256):
        _,c=parent();c['parameter_endpoints'][1]='123'
        with pytest.raises(ValueError):wider_inputs(c)
        _,c=parent();c['passed']=False
        with pytest.raises(ValueError):wider_inputs(c)
