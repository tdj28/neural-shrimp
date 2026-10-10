from fractions import Fraction
import copy
import numpy as np
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_boundary_diagnostics import fold_indicator
from neural_shrimp.rigorous_eigenpair_radius_forecast import march as forecast_march,radius_settings
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube


def march(model,parent,**kwargs):
    return forecast_march(model,parent,['0']*len(parent['weights']),**kwargs)


def test_nonnormal_coupled_fold_proxy_has_correct_left_and_composition():
    with ctx.workprec(256):
        _,c=parent();events=[];d=fold_indicator(c,lambda k,v:events.append(k))
        assert events==['fold_indicator_linear_return']
        assert d['diagnostic_residuals_small'] and abs(d['left_mode_pairing']-1)<1e-14
        assert abs(d['second_derivative_contraction']+20.25)<1e-12
        np.testing.assert_allclose(d['a_actual_rates'],[1.5,-.5],atol=1e-14)
        assert d['quadratic_coefficient']==d['second_derivative_contraction']/2


def test_four_fresh_native_charts_no_repair():
    with ctx.workprec(256):
        m,c=parent();ticks=[];events=[];r=march(m,c,tick=lambda:ticks.append(1),observe=lambda k,v:events.append(k))
        assert r['completed'] and r['calls']==4 and r['repairs']==0 and len(ticks)==60
        assert events.count('march_accepted')==4 and events.count('fold_indicator_linear_return')==4


def test_one_repair_per_chart_preserves_failed_attempts_and_floor():
    with ctx.workprec(256):
        m,c=parent();calls=[];events=[]
        def tube(*a,**kw):
            calls.append(kw);r,d=quadratic_tube(*a,**kw)
            if len(calls)%2:r['passed']=False;r['gates']['uniform_normalized_eigenpair_locus']=False
            return r,d
        def proposer(c,observe):
            old=[Fraction(w)*Fraction(2)**c['settings']['radius_exponent'] for w in c['weights']]
            return {'old_exact_radii':list(map(str,old)),'exact_radii':[str(v*Fraction(1001,1000)) for v in old]}
        r=march(m,c,tube=tube,proposer=proposer,observe=lambda k,v:events.append((k,v)))
        assert r['completed'] and r['calls']==8 and r['repairs']==4 and len(r['attempts'])==8
        for i in range(0,8,2):
            assert calls[i]['parameter']==calls[i+1]['parameter']
            old=[Fraction(v)*Fraction(2)**calls[i]['radius_exponent'] for v in calls[i]['weights']]
            new=[Fraction(v)*Fraction(2)**calls[i+1]['radius_exponent'] for v in calls[i+1]['weights']]
            assert all(b>=a for a,b in zip(old,new))
        assert len([1 for k,v in events if k=='march_radius_rounding'])==4


def test_second_failure_cannot_advance_last_chart():
    with ctx.workprec(256):
        m,c=parent()
        def tube(*a,**kw):
            r,d=quadratic_tube(*a,**kw);r['passed']=False;r['gates']['uniform_normalized_eigenpair_locus']=False;return r,d
        def proposer(c,o):
            old=[str(Fraction(w)*Fraction(2)**c['settings']['radius_exponent']) for w in c['weights']];return {'old_exact_radii':old,'exact_radii':old}
        r=march(m,c,tube=tube,proposer=proposer)
        assert not r['completed'] and r['calls']==2 and r['last_chart']['parameter_endpoints']==c['parameter_endpoints'] and not r['accepted']


def test_failed_accepted_durability_stops_later_calls():
    with ctx.workprec(256):
        m,c=parent();ticks=[]
        def obs(k,v):
            if k=='march_accepted':raise OSError('durability')
        with pytest.raises(OSError):march(m,c,tick=lambda:ticks.append(1),observe=obs)
        assert len(ticks)==15


def test_diagnostic_store_valueerror_is_not_numeric_abstention():
    from neural_shrimp.rigorous_eigenpair_radius_forecast import DiagnosticObserverFailure
    with ctx.workprec(256):
        m,c=parent();ticks=[];events=[]
        def obs(k,v):
            events.append(k)
            if k=='midpoint_eig_return':raise ValueError('aggregate raw cap')
        with pytest.raises(DiagnosticObserverFailure,match='aggregate raw cap'):
            march(m,c,tick=lambda:ticks.append(1),observe=obs)
        assert len(ticks)==15 and 'march_accepted' not in events and 'fold_indicator_linear_return' not in events


def driver():
    import sys,importlib
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
    return importlib.import_module('continue_eigenpair_radius_forecast')


def test_fixed_increment_never_amplifies_from_its_own_forecast():
    with ctx.workprec(256):
        m,c=parent();old=[Fraction(w)*Fraction(2)**c['settings']['radius_exponent'] for w in c['weights']]
        delta=[v/256 for v in old];events=[]
        r=forecast_march(m,c,list(map(str,delta)),observe=lambda k,v:events.append((k,v)))
        assert r['completed'] and r['calls']==4 and r['repairs']==0
        forecasts=[v for k,v in events if k=='initial_radius_forecast'];assert len(forecasts)==4
        for k,v in enumerate(forecasts):
            assert list(map(Fraction,v['fixed_increment']))==delta
            actual=list(map(Fraction,v['settings']['exact_radii']))
            assert actual==[a+2*(k+1)*d for a,d in zip(old,delta)]
        assert all(a['gates']['anchor_identified'] for a in r['accepted'])


def test_forecast_durability_failure_is_before_any_fresh_model_call():
    with ctx.workprec(256):
        m,c=parent();ticks=[]
        def obs(k,v):
            if k=='initial_radius_forecast':raise ValueError('cap')
        with pytest.raises(ValueError,match='cap'):forecast_march(m,c,['0']*len(c['weights']),tick=lambda:ticks.append(1),observe=obs)
        assert ticks==[]


@pytest.mark.parametrize('change',['negative','oversize','shape'])
def test_forecast_rejects_bad_proposal_before_any_fresh_model_call(change):
    with ctx.workprec(256):
        m,c=parent();d=['0']*len(c['weights']);ticks=[]
        if change=='negative':d[0]='-1'
        if change=='oversize':d[0]='1'
        if change=='shape':d.pop()
        with pytest.raises(ValueError):forecast_march(m,c,d,tick=lambda:ticks.append(1))
        assert ticks==[]


def history_fixture():
    with ctx.workprec(256):_,c=parent()
    c=copy.deepcopy(c);c['settings']['halfwidth_exponent']=-6;c['settings']['radius_exponent']=-16
    charts=[]
    for j in range(5):
        v=copy.deepcopy(c);v['weights']=[str(Fraction(1,2)+Fraction(j*(i%2+1),1024)) for i in range(len(v['weights']))]
        v['parameter_endpoints']=[str(Fraction(51+4*j,128)),str(Fraction(55+4*j,128))];charts.append(v)
    return charts


def test_history_uses_exact_fixed_componentwise_max_and_consecutive_native_join():
    h=history_fixture();d=driver();v=d.history_increment(h)
    assert list(map(Fraction,v['fixed_increment']))==[Fraction(i%2+1,1024*2**16) for i in range(len(h[0]['weights']))]
    assert v['multiplier']==2 and v['relative_increment_cap']=='1/16'


@pytest.mark.parametrize('change',['negative','decrease','native','endpoint','length'])
def test_historical_forecast_fails_closed(change):
    h=history_fixture();d=driver()
    if change=='negative':h[1]['weights'][0]='-1'
    if change=='decrease':h[1]['weights'][0]='1/4'
    if change=='native':h[2]['sigma']=-1
    if change=='endpoint':h[2]['parameter_endpoints'][0]='-1'
    if change=='length':h.pop()
    with pytest.raises(ValueError):d.history_increment(h)


def test_fixed_driver_pins_and_complete_local_import_inventory():
    from argparse import Namespace
    d=driver();a=Namespace(unit=4,expected_right='71/128',parent=d.T.BASE/'attempt-003-wider-march',**d.PINS)
    d.install(a);a.parent_source='0'*40
    with pytest.raises(ValueError,match='fixed complete'):d.install(a)
    assert set(d.C.SOURCES)<set(d.SOURCES)
    for n in d.EXTRA:assert (d.ROOT/n).is_file()
