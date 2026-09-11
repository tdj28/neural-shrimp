"""Instrument fixtures only: no target neural dynamics or grid outcomes."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('hopfield_driver',ROOT/'tools/run_hopfield.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
PLAN=json.loads((ROOT/'experiments/hopfield/EXP-004-plan.json').read_text())

def test_recurrence_uses_all_section_coordinates_and_rejects_scalar_alias():
    base=np.array([[1.,0.,2.],[-1.,1.,2.],[.4,-1.,2.]])
    event=np.column_stack([np.arange(24.),np.tile(base,(8,1))])
    stats=np.zeros(20);stats[16]=1
    q,residual,duration=driver.classify(event,stats,PLAN)
    assert q==3 and residual==0 and duration==3
    event[:,1]+=np.arange(24)*.01
    assert driver.classify(event,stats,PLAN)[0]==0

def test_short_tail_and_equilibrium_do_not_create_nontrivial_period():
    stats=np.zeros(20);stats[16]=1
    events=np.array([[0,1,2,3],[1,2,3,4],[2,1,2,3]],float)
    assert driver.classify(events,stats,PLAN)[0]==0
    assert driver.classify(np.tile(events,(4,1)),np.zeros(20),PLAN)[0]==0

def test_plan_count_and_missing_core_are_explicit():
    assert driver.validate(PLAN)==39523
    g={'alpha':np.tile(np.linspace(1.056,1.08,3),(3,1)), 'gamma':np.linspace(10.14,10.55,3), 'period':np.zeros((3,3),int), 'stats':np.zeros((3,3,20))}
    assert all(value is None for value in driver.select_points(g,PLAN).values())

def test_partial_row_retains_each_completed_full_cell(tmp_path,monkeypatch):
    def fixture_cell(fn,alpha,gamma,initial,p,**kw):
        if alpha==3:raise RuntimeError('fixture interruption')
        return np.array([alpha,2.,3.]),np.zeros((128,4)),np.zeros(20),8,2,.01,4.
    monkeypatch.setattr(driver,'cell',fixture_cell)
    path=tmp_path/'row-000.npz'
    with pytest.raises(RuntimeError,match='fixture interruption'):
        driver.row(None,[1.,2.,3.],10.,PLAN,path,time.monotonic()+10)
    assert not path.exists()
    with path.with_suffix('.journal').open('rb') as stream:
        first=np.load(stream,allow_pickle=False);second=np.load(stream,allow_pickle=False)
        assert len(stream.read())==0
    assert first[0]==1 and second[0]==2
    assert np.array_equal(second[2:5],first[5:8])
    assert json.loads(path.with_suffix('.failure.json').read_text())['completed_cells']==2
