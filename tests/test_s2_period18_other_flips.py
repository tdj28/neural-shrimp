"""Selection and bounded-driver controls; no consumed target arrays."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('other_flips',ROOT/'tools/explore_s2_period18_other_flips.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_fixed_selection_excludes13_and_preserves_full_census():
    window={'center':[0.,0.],'scale':[1.,1.],'period':18,'rays':[{'type':'flip','s_boundary':.1,'direction':[1.,0.]} for _ in range(16)]}
    saved={'period':np.array([[18]*3,[36]*3,[36]*3]),'status':np.full((3,3),2),'log10_hidden':np.array([0.,.5,1.]),'log10_output':np.array([0.,.5,1.])}
    out=m.select_rays(window,saved);assert out['selected']==[0,1] and len(out['all_flip_rows'])==16
    window['rays'][:13]=[{'type':'fold','s_boundary':.1,'direction':[1.,0.]} for _ in range(13)]
    assert m.select_rays(window,saved)['selected']==[14,15]

def test_nonperiodic_first_cell_is_not_adjacent36():
    window={'center':[0.,0.],'scale':[1.,1.],'period':18,'rays':[{'type':'flip','s_boundary':.1,'direction':[1.,0.]}]*2}
    saved={'period':np.full((3,3),36),'status':np.ones((3,3),int),'log10_hidden':np.array([0.,.5,1.]),'log10_output':np.array([0.,.5,1.])}
    with pytest.raises(ValueError):m.select_rays(window,saved)

def test_shared_budget_honors_previous_ray_calls(tmp_path):
    m.ROOT_OUT=tmp_path;e=object.__new__(m.Experiment);e.budget={'stopped':False,'start':m.time.monotonic(),'completed_points':m.CALLS-1};e.points=1
    with pytest.raises(TimeoutError):e.check()

def test_finalization_reserve_survives_ordinary_limit(tmp_path,monkeypatch):
    m.ROOT_OUT=tmp_path;monkeypatch.setattr(m,'ORDINARY',1)
    with pytest.raises(OSError):m.save(tmp_path/'too-big.json',{'data':2})
    m.save(tmp_path/'failure.json',{'retained':True},final=True)
    assert (tmp_path/'failure.json').exists()


def test_signed_rate_derivative_agrees_with_finite_difference(tmp_path):
    from fractions import Fraction
    m.ROOT_OUT=tmp_path
    inp={'consumed_map':{'network':{'d':1,'width':1,'activation':'tanh','bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[['1/3'],['-1/2']],'y':['1/4','-1/5']},'family':{'ell_float':[-1.,-1.3],'direction_float':[-.2,.3]}}
    exp=m.Experiment(inp,tmp_path,{'start':m.time.monotonic(),'completed_points':0,'stopped':False})
    try:
        x=np.array([.1,.2,-.1,.05]);s=.2;eps=1e-5
        actual=exp.evaluate(x,s,3);plus=exp.evaluate(x,s+eps,3);minus=exp.evaluate(x,s-eps,3)
        fd=(np.array(plus['orbit'][-1])-np.array(minus['orbit'][-1]))/(2*eps)
        assert actual['rate_derivative'][0]<0<actual['rate_derivative'][-1]
        np.testing.assert_allclose(actual['parameter_derivative'],fd,rtol=2e-7,atol=1e-10)
        with pytest.raises(ValueError):exp.net.rate_vector([-.1],[.2])
    finally:exp.close()
