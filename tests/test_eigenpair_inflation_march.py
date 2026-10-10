from fractions import Fraction
import copy
import numpy as np
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_boundary_diagnostics import fold_indicator
from neural_shrimp.rigorous_eigenpair_inflation_march import march,radius_settings
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube


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
