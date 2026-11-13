import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('audit_fold_return_line',Path(__file__).parents[1]/'tools/audit_fold_return_line.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_independent_visits_match_known_two_phase_sequence():
    mask=np.zeros(40,dtype=bool);mask[2:12:2]=1;mask[19:29:2]=1;mask[34:38:2]=1
    visits=m.visits_from_mask(mask)
    assert [(v['entry'],v['last_inside'],v['phase']) for v in visits]==[(2,10,0),(19,27,1)]


def test_capture_requires_persistent_order_and_ignores_one_close_pass():
    orbit=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    x=np.concatenate([np.ones((10,2))*4,orbit[np.arange(200)%3]])
    record=m.capture_trace(x,orbit)
    assert record['first_128_match_start'] in (10,11)
    assert record['last128_ordered'] and record['last128_max_relative_distance']==0
    x=orbit[-np.arange(200)%3]
    assert m.capture_trace(x,orbit)['first_128_match_start'] is None


def test_interpolation_orders_coordinates_and_averages_duplicates():
    actual=m.interpolate(np.array([2.,0.,1.,1.]),np.array([[4.],[0.],[1.],[3.]]),np.array([.5,1.5]))
    np.testing.assert_allclose(actual,[[1.],[3.]])


def test_grouped_error_does_not_overweight_a_trajectory_with_many_visits():
    metrics=m.grouped_metrics(np.zeros((4,1)),np.array([[1.],[1.],[1.],[3.]]),np.array([0,0,0,1]))
    assert np.isclose(metrics['RMSE'][0],np.sqrt(3))
    assert np.isclose(metrics['equal_row_RMSE'][0],np.sqrt(5))
