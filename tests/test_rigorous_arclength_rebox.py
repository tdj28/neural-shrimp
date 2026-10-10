import copy,importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import matrix_json
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_arclength_rebox import rebox,proposed_radii,stream_reboxed
from test_rigorous_arclength_connection import Fold,inputs
from test_rigorous_arclength_batch import FakeDerivative


def full():return two_arclength_charts(Fold(),Fold(),inputs())['charts'][-1]


def loose_failure():
    c=full();c['predictor_residual']=matrix_json(arb_mat([[arb(0,arb('1/65536'))],[0]]));c['gates']['uniform_root']=False;c['passed']=False
    return c


def test_loose_residual_box_is_reacquired_and_proved():
    class Recording(Fold):
        def __init__(self):self.calls=[]
        def step(self,x,h):self.calls.append((copy.copy(x),h));return super().step(x,h)
    with ctx.workprec(256):
        initial=loose_failure();before=copy.deepcopy(initial);proposal=proposed_radii(initial);m=Recording()
        r=rebox(m,initial,proposal)
        assert r['passed'] and len(m.calls)==1 and initial==before
        assert r['settings']['radius_exponent']>initial['settings']['radius_exponent']
        assert r['predictor_residual']==initial['predictor_residual']
        assert r['jacobian_tube']!=initial['jacobian_tube']
        assert r['anchor']['state_box']==initial['anchor']['state_box']


def test_changed_derivative_is_not_ignored():
    class Wrong(Fold):
        def step(self,x,h):return [x[0]],arb_mat([[10**20]]),[arb(-1)]
    c=loose_failure();r=rebox(Wrong(),c,proposed_radii(c))
    assert not r['gates']['uniform_root']


def test_proposal_cap_and_nonpositive_weights_fail_closed():
    c=loose_failure();p=proposed_radii(c)
    p['exact_weights'][0]='0'
    with pytest.raises(ValueError,match='positive exact'):rebox(Fold(),c,p)
    p=proposed_radii(c);p['radius_exponent']=-9
    with pytest.raises(ValueError,match='radius bound'):rebox(Fold(),c,p)


def test_profile_forces_full_reboxing_paths_without_scientific_override():
    old=compact_chart(full());initial=[];final=[]
    r=stream_reboxed(Fold(),Fold(),old,count=2,synthetic_profile=True,force_profile_rebox=True,
        emit_initial=lambda i,c:initial.append(copy.deepcopy(c)),emit_chart=lambda i,c:final.append(copy.deepcopy(c)))
    assert r['passed'] and r['adapted_charts']==2 and len(initial)==len(final)==2
    assert all(c['reboxing']['fresh_whole_domain_jacobian'] for c in final)
    with pytest.raises(ValueError,match='engineering'):stream_reboxed(Fold(),Fold(),old,force_profile_rebox=True)


def test_initial_failure_persisted_before_proposal_and_only_one_rebox(monkeypatch):
    import neural_shrimp.rigorous_arclength_rebox as m
    c=loose_failure();events=[]
    monkeypatch.setattr(m,'arclength_tube',lambda *a,**k:copy.deepcopy(c))
    def rejected(*a,**k):events.append('proposal');raise ValueError('synthetic invalid proposal')
    monkeypatch.setattr(m,'proposed_radii',rejected)
    with pytest.raises(ValueError,match='invalid proposal'):
        m.stream_reboxed(Fold(),Fold(),compact_chart(full()),count=3,emit_initial=lambda i,r:events.append('durable initial'))
    assert events==['durable initial','proposal']


def test_failed_rebox_stops_and_cannot_anchor(monkeypatch):
    import neural_shrimp.rigorous_arclength_rebox as m
    c=loose_failure();calls=[]
    monkeypatch.setattr(m,'arclength_tube',lambda *a,**k:copy.deepcopy(c))
    def no(*a,**k):
        calls.append(1);r=copy.deepcopy(c);r['reboxing']={};return r
    monkeypatch.setattr(m,'rebox',no)
    r=m.stream_reboxed(Fold(),Fold(),compact_chart(full()),count=3)
    assert calls==[1] and r['completed_charts']==1 and r['qualified_charts']==0


def test_primitive_or_anchor_failure_never_triggers_radius_retry(monkeypatch):
    import neural_shrimp.rigorous_arclength_rebox as m
    for gate in ('primitive_period','anchor_identified'):
        c=loose_failure();c['gates'][gate]=False
        monkeypatch.setattr(m,'arclength_tube',lambda *a,**k:copy.deepcopy(c))
        def forbidden(*a,**k):raise AssertionError('unexpected proposal')
        monkeypatch.setattr(m,'proposed_radii',forbidden)
        assert not m.stream_reboxed(Fold(),Fold(),compact_chart(full()),count=3)['passed']


def test_full1440_counter_rotation_and_caps():
    from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
    with ctx.workprec(256):
        m=ReboxingMeanValueHiddenRateMap({},1440,factory=FakeDerivative)
        for _ in range(1440):m.step([],arb(1))
        assert m.instances==20 and m.counts()==dict(map_attempted=1440,map_returned=1440,unit_attempted=2880,unit_returned=2880,map_cap=1440,unit_cap=2880)
        with pytest.raises(ValueError,match='aggregate'):m.step([],arb(1))
        with pytest.raises(ValueError,match='aggregate'):ReboxingMeanValueHiddenRateMap({},1441,factory=FakeDerivative)


def driver():
    s=importlib.util.spec_from_file_location('reboxdriver',Path(__file__).parents[1]/'tools/certify_s2_period18_arclength_reboxing.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_new_proof_profile_sources_and_partial_counts_bound():
    m=driver()
    for path in ('src/neural_shrimp/rigorous_arclength_rebox.py','src/neural_shrimp/rigorous_reboxing_gradient.py','tools/certify_s2_period18_arclength_reboxing.py','tests/test_rigorous_arclength_rebox.py'):
        assert path in m.SOURCES
    assert m.PARTIAL_MANIFEST=='26cf51439d0d409dbe8b9d03c5b774805d5c93f836d3a3045681f3c065dc3135'


def test_profile_checks_new_complete_reboxing_budget():
    m=driver();rt={'python':m.platform.python_version(),'python_flint':m.flint.__version__,'flint':m.flint.__FLINT_VERSION__,
      'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'mode':'profile','threads':dict.fromkeys(m.THREADS,'1'),'nice':10,
      'internal_seconds':20,'tick_cap':400,'map_cap':180,'unit_derivative_cap':360,'jet_cap':108}
    m.profile_environment(rt,{'elapsed_seconds':1,'max_rss_kib':1,'exit_code':0})
    rt['map_cap']=144
    with pytest.raises(ValueError,match='library/settings'):m.profile_environment(rt,{'elapsed_seconds':1,'max_rss_kib':1,'exit_code':0})


def test_durable_fold_membership_stops_before_next_chart():
    saved=[]
    def emit(i,c):
        c['known_fold_membership']={'all_cyclic_alignments':[{'contained_in_qualified_chart':True}]}
        saved.append(copy.deepcopy(c));return {'durable':True}
    r=stream_reboxed(Fold(),Fold(),compact_chart(full()),count=3,emit_chart=emit)
    assert r['passed'] and r['known_fold_identified'] and r['completed_charts']==r['qualified_charts']==1 and len(saved)==1
