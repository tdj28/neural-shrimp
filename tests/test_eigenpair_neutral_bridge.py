import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import pytest
from flint import ctx
from neural_shrimp.eigenpair_boundary_target import event_flags,BoundaryTargetReached
from neural_shrimp.rigorous_eigenpair_inflation_march import march
from test_eigenpair_reanchor import parent


def diagnostics(other=.9,c=-20):
    return {'spectrum':{'eigenvalues':[[1.,0.]]+[[other,0.]]+[[.1+i*.03,0.]for i in range(11)],'critical_nearest_index':0,'relative_residuals':[0.]*13,'mode_residual':0.},'fold':{'second_derivative_contraction':c}}


def test_neutral_mode_excluded_but_competing_events_stop():
    assert event_flags(diagnostics())==[]
    assert 'nontrivial_plus_one_gap_at_most_0.03'in event_flags(diagnostics(.98))
    assert 'nontrivial_unit_circle_margin_at_most_0.03'in event_flags(diagnostics(-.98))
    assert 'fold_coefficient_proxy_reaches_minus_one_or_above'in event_flags(diagnostics(c=-.5))
    d=diagnostics();d['spectrum']['eigenvalues'][1]=[.7,.7]
    assert 'nontrivial_unit_circle_margin_at_most_0.03'in event_flags(d)


def test_unresolved_mode_is_retained_stop_not_silent_continue():
    d=diagnostics();d['spectrum']['relative_residuals'][1]=1e-4
    assert event_flags(d)==['saved_mode_residual_unresolved']
    assert event_flags({'spectrum':{'unresolved':'bad'}})==['saved_diagnostics_unresolved']


def test_event_after_durable_accepted_root_prevents_another_tube():
    with ctx.workprec(256):
        m,c=parent();ticks=[];events=[]
        def observe(k,v):
            events.append(k)
            if k=='march_accepted':raise BoundaryTargetReached(event_flags(diagnostics(.98)))
        with pytest.raises(BoundaryTargetReached):march(m,c,tick=lambda:ticks.append(1),observe=observe)
        assert len(ticks)==15 and events[-1]=='march_accepted' and events.count('chart_return')==1


def driver():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_neutral_bridge.py';spec=importlib.util.spec_from_file_location('neutral_bridge_driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def args(m,unit=1):
    return SimpleNamespace(unit=unit,expected_right=str(Fraction(227+8*(unit-1),128)),parent=m.T.BASE/('attempt-001-narrower-target' if unit==1 else f'attempt-{unit-1:03d}-neutral-bridge'),**m.FIRST_PINS)


def test_exact_first_and_final_fixed_bridge_selection():
    m=driver();m.install(args(m));m.install(args(m,9))
    assert m.march is march and set(m.N.SOURCES)<=set(m.SOURCES)
    assert all((Path(__file__).parents[1]/n).is_file()for n in m.SOURCES)


@pytest.mark.parametrize('defect',['unit','path','source','endpoint'])
def test_invalid_bridge_before_acquisition(defect):
    m=driver();a=args(m)
    if defect=='unit':a.unit=10
    if defect=='path':a.parent=Path('/tmp/other')
    if defect=='source':a.parent_source='b'*40
    if defect=='endpoint':a.expected_right='228/128'
    with pytest.raises(ValueError):m.install(a)


def test_larger_reference_admission_fails_closed():
    m=driver();old={'passed':True,'ticks':254,'status':'two_narrower_charts','elapsed_seconds':36.925,'stored_bytes_before_final':86368732,'peak_rss_kib':1125524}
    assert m.cost_gate(old)['seconds']<400
    old['elapsed_seconds']=70
    with pytest.raises(ValueError):m.cost_gate(old)
