import copy
import numpy as np
import pytest
from neural_shrimp.tiny_lm_horizon import predict,evaluate,summarize,first_crossing


def test_fixed_growth_warning_vs_persistent_defect():
    u=np.array([1.,0]);p=predict(u,u,u,np.array([.95,0]),u,u,1.,1.)
    assert p['predicted_crossing']['drift']==4
    assert p['predicted_crossing']['persistent_defect']is None
    assert p['scores']['drift']==pytest.approx([0,.05,.15,.3])


def test_constant_linear_map_exact_response_and_no_warning():
    A=np.diag([.99,.8]);u=np.array([1.,0]);series=np.array([np.linalg.matrix_power(A,h)@u for h in range(5)])
    p=predict(u,series[1],series[2],series[2],u,u,1.,1.)
    e=evaluate(p,u,series,series,series,series)
    assert e['resolved']and e['right_censored']and e['future_warning_eligible']
    assert not e['warnings']['drift']and not e['future_failure']


def test_later_data_cannot_change_early_forecast_and_unresolved_retained():
    u=np.array([1.,0]);early=(u,u,u,np.array([.95,0]),u,u,1.,1.)
    p=predict(*early);before=copy.deepcopy(p);f=np.array([u]*5);ordered=f.copy();small=f.copy();big=f.copy()
    small[4]*=2;big[4]*=2;ordered[4]*=2
    e=evaluate(p,u,f,ordered,big,small)
    assert p==before and e['future_failure']and e['observed_crossing']==4
    big[3]*=2;assert not evaluate(p,u,f,ordered,big,small)['resolved']
    with pytest.raises(ValueError):first_crossing([.1]*3)


def test_existing_failure_is_detection_not_future_warning_and_counts():
    u=np.array([1.,0]);p=predict(u,u,u,np.array([.95,0]),u,u,1.,1.)
    small=np.array([u]*5);f=small.copy();f[2]*=2
    e=evaluate(p,u,f,small,small,small)
    assert e['resolved']and not e['future_warning_eligible']
    r=summarize([dict(evaluation=e)]*32)
    assert r['already_failing_rows']==32 and r['future_eligible_rows']==0
    assert all(sum(c.values())==0 for c in r['future_warning_counts'].values())
