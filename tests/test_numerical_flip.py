import numpy as np
import pytest
from neural_shrimp.numerical_flip import correct,qualify,refine_flip,crossing,doubled


def evaluator(x,s,period):
    # Supercritical flip at s=0; fixed-point branch itself moves with s.
    x=np.asarray(x);z=x.copy();m=np.eye(2);b=np.zeros(2);orbit=[z.tolist()];check={}
    for k in range(period):
        u=z[0]-s
        j=np.diag([-1-s+3*u*u,.5])
        fs=np.array([2+s-u-3*u*u,0.])
        b=j@b+fs;m=j@m
        z=np.array([s-(1+s)*u+u**3,.5*z[1]])
        orbit.append(z.tolist());check[str(k+1)]={'state':z.tolist(),'monodromy':m.tolist(),'parameter_derivative':b.tolist()}
    return {'period':period,'orbit':orbit,'monodromy':m.tolist(),'parameter_derivative':b.tolist(),'residual':(z-x).tolist(),'checkpoints':check}


def test_corrector_and_primitive_rejection():
    c=correct(evaluator,[.011,.02],.01,1)
    assert c['qualified'] and np.max(abs(np.array(c['state'])-[.01,0]))<1e-12
    assert not qualify(evaluator([0,0],0,2))['gates']['primitive']


def test_flip_and_total_crossing_on_moving_branch():
    p=refine_flip(evaluator,np.array([-.001,0.]),-.001,1)
    assert p['qualified'] and abs(p['s'])<1e-10
    c=crossing(evaluator,p,1)
    assert c['qualified'] and abs(c['series'][-1]['corrected_branch_crossing']+1)<1e-7
    assert np.max(abs(np.array(c['tangent'])-[1,0]))<1e-9


def test_amplitude_constraint_finds_noncollapsed_doubled_cycle():
    p=refine_flip(evaluator,np.array([-.001,0.]),-.001,1)
    r=doubled(evaluator,p,1,.002)
    assert r['qualified'] and r['qualification']['stability']=='attracting'
    assert abs(r['s']-4e-6)<1e-10 and r['phase_separation']>.0039


def test_wrong_multiplier_rejected():
    def wrong(x,s,p):
        r=evaluator(x,s,p);r['monodromy']=(.5*np.eye(2)).tolist();return r
    p=refine_flip(wrong,np.zeros(2),0.,1)
    assert not p['qualified']


def test_spectrum_residual_gate_catches_tamper(monkeypatch):
    import neural_shrimp.numerical_flip as m
    monkeypatch.setattr(m,'spectrum',lambda _: (np.array([.5,.5]),np.eye(2),[.1,.1]))
    assert not qualify(evaluator([0,0],0,1))['qualified']


def test_subcritical_doubled_orbit_is_primitive_but_unstable():
    def sub(x,s,period):
        x=np.asarray(x);z=x.copy();m=np.eye(2);b=np.zeros(2);orbit=[z.tolist()];check={}
        for k in range(period):
            j=np.diag([-1-s-3*z[0]**2,.5]);fs=np.array([-z[0],0.]);b=j@b+fs;m=j@m
            z=np.array([-(1+s)*z[0]-z[0]**3,.5*z[1]])
            orbit.append(z.tolist());check[str(k+1)]={'state':z.tolist(),'monodromy':m.tolist(),'parameter_derivative':b.tolist()}
        return {'period':period,'orbit':orbit,'monodromy':m.tolist(),'parameter_derivative':b.tolist(),'residual':(z-x).tolist(),'checkpoints':check}
    p=refine_flip(sub,np.zeros(2),-.001,1);r=doubled(sub,p,1,.002)
    assert r['qualified'] and r['qualification']['stability']=='unstable'
    assert abs(r['s']+4e-6)<1e-10


def test_doubled_parent_collapse_is_rejected():
    p=refine_flip(evaluator,np.array([-.001,0.]),-.001,1)
    r=doubled(evaluator,p,1,0.)
    assert not r['qualified'] and not r['gates']['primitive_doubled'] and not r['gates']['noncollapsed']


def test_aggregate_limit_preserves_final_failure(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('flip_driver_control',Path(__file__).resolve().parents[1]/'tools/explore_s2_period18_flip.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'ARTIFACT_ROOT',tmp_path)
    monkeypatch.setattr(mod,'total_bytes',lambda:28*1024**2)
    with pytest.raises(OSError,match='aggregate artifact limit'):mod.save(tmp_path/'path.json',{'returned':'duplicate'})
    assert (tmp_path/'oversized-artifact.json').exists()
    mod.save(tmp_path/'failure.json',{'failed':True},final=True)
    assert (tmp_path/'failure.json').exists() and not (tmp_path/'path.json').exists()
