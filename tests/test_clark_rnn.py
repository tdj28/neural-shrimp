import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.clark_rnn import inputs,step,jacobian,iterate,period_proposals,qualify_cycle,simulate_batch,divisors

def test_seed_and_protected_guard():
    assert all(np.array_equal(a,b) for a,b in zip(inputs(4,42),inputs(4,42)))
    assert not np.array_equal(inputs(4,42)[0],inputs(4,43)[0])
    with pytest.raises(ValueError):inputs(4,29001)

def test_odd_map_even_jacobian_and_origin():
    j,x,_=inputs(4,42);z=x[0]
    np.testing.assert_array_equal(step(-z,j,2,.7),-step(z,j,2,.7))
    np.testing.assert_array_equal(jacobian(-z,j,2,.7),jacobian(z,j,2,.7))
    np.testing.assert_allclose(jacobian(np.zeros(4),j,2,.7),.3*np.eye(4)+1.4*j,atol=1e-15)

def test_jacobian_and_chain_rule():
    j,x,_=inputs(3,45);z=x[0];eps=1e-5
    fd=np.column_stack([(step(z+eps*e,j,1.7,.4)-step(z-eps*e,j,1.7,.4))/(2*eps) for e in np.eye(3)])
    np.testing.assert_allclose(fd,jacobian(z,j,1.7,.4),atol=2e-10,rtol=1e-9)
    y,m,states,js=iterate(z,j,1.7,.4,4)
    fd=np.column_stack([(iterate(z+eps*e,j,1.7,.4,4)[0]-iterate(z-eps*e,j,1.7,.4,4)[0])/(2*eps) for e in np.eye(3)])
    np.testing.assert_allclose(m,fd,atol=2e-10,rtol=1e-8)

def test_batch_matches_scalar_and_physical_time():
    j=np.array([[.5]]);x=np.zeros((2,1));v=np.ones((2,1))
    a=simulate_batch(j,x,v,np.array([1.]),np.array([.5]),burn=0,align=0,measured=1024)
    np.testing.assert_allclose(a['lambda_per_step'],np.log(.75),atol=2e-15)
    np.testing.assert_allclose(a['lambda_per_time'],np.log(.75)/.5,atol=4e-15)
    np.testing.assert_array_equal(a['period_proposals'],[1,1])

def p4():
    a=2.
    for _ in range(15):a-=(a-2*np.tanh(a))/(1-2*(1-np.tanh(a)**2))
    return np.array([[0.,-2.],[2.,0.]]),np.array([a,a])

def test_primitive_p4_and_odd_self_pair():
    j,x=p4();out=qualify_cycle(x+1e-5,j,1.,1.,4)
    assert out['status']=='numerical_attracting_primitive'
    assert out['odd_self_pair_distance']<1e-10 and out['radius']<.2
    assert out['proper_divisor_separations']['2']>1
    assert qualify_cycle(x,j,1.,1.,8)['status']=='closed_unqualified'

def test_full_state_not_even_observable_period():
    j,x=p4();cycle=iterate(x,j,1.,1.,4)[2]
    tail=np.tile(cycle,(129,1))[:,None,:]
    p,_=period_proposals(tail);assert p[0]==4
    assert np.ptp(np.sum(tail**2,axis=2))<1e-12

def test_nonclosing_tail_and_insufficient_tail():
    rng=np.random.default_rng(51);p,_=period_proposals(rng.normal(size=(513,3,2)));assert np.all(p==0)
    with pytest.raises(ValueError):period_proposals(np.zeros((20,1,2)))

def test_positive_tangent_synthetic_expansion():
    out=simulate_batch(np.array([[2.]]),np.zeros((2,1)),np.ones((2,1)),np.array([1.]),np.array([1.]),burn=0,align=0,measured=1024)
    np.testing.assert_allclose(out['lambda_per_step'],np.log(2),atol=2e-15)
    assert np.all(out['period_proposals']==1) # unstable fixed trajectory is not chaos

def test_interruption_retains_current_checkpoint():
    calls=[];checks=[0]
    def check():
        checks[0]+=1
        if checks[0]==2:raise TimeoutError('test')
    with pytest.raises(TimeoutError):simulate_batch(np.array([[.5]]),np.zeros((2,1)),np.ones((2,1)),np.array([1.]),np.array([.5]),burn=0,align=0,measured=1024,check=check,checkpoint=lambda t,x,v:calls.append((t,x.copy(),v.copy())))
    assert calls[-1][0]==128

def test_profile_rejects_nonfinite_timing(tmp_path):
    spec=importlib.util.spec_from_file_location('clark_runner',Path(__file__).resolve().parents[1]/'tools/explore_clark_rnn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    identity={'sha256':{'a':'b'}}
    m.save(tmp_path/'runtime-start.json',{'mode':'profile','source':identity,'numpy':np.__version__,'python':m.sys.version,'platform':m.platform.platform(),'thread_environment':{'OMP_NUM_THREADS':'1'},'nice':10,'mem_available_bytes':2**30})
    for bad in [float('nan'),float('inf'),-1.,0.]:
        (tmp_path/'result.json').write_text(m.json.dumps({'status':'completed','batch_seconds':bad,'period64_unit_seconds':.001,'elapsed_seconds':1,'peak_rss_kib':50000}))
        m.manifest(tmp_path)
        with pytest.raises(ValueError):m.profile_gate(tmp_path,m.sha(tmp_path/'manifest.json'),identity)

def test_scalar_monotone_restriction_and_divisors():
    # N1, delta1: a*tanh is increasing for a>0 and decreasing for a<0.
    x=np.linspace(-4,4,41);assert np.all(np.diff(2*np.tanh(x))>0)
    assert np.all(np.diff(-2*np.tanh(x))<0)
    assert divisors(12)==[1,2,3,4,6]


def test_axis_cycle_transverse_instability_not_hidden():
    j,x=p4();x[1]=0
    out=qualify_cycle(x,j,1.,1.,4)
    assert out['status']=='closed_unqualified' and out['radius']>15.9


def test_annihilated_tangent_is_retained_unresolved():
    a=simulate_batch(np.zeros((1,1)),np.zeros((2,1)),np.ones((2,1)),np.array([1.]),np.array([1.]),burn=0,align=0,measured=1024)
    assert np.all(a['tangent_collapse_count']==1024)
    assert np.all(np.isnan(a['lambda_per_step']))
    assert np.all(a['period_proposals']==1)


def test_returned_iterates_survive_later_failure(monkeypatch):
    import neural_shrimp.clark_rnn as core
    j,x=p4();events=[];original=core.iterate;calls=[0]
    def damaged(*args):
        calls[0]+=1
        if calls[0]==2:raise FloatingPointError('synthetic later failure')
        return original(*args)
    monkeypatch.setattr(core,'iterate',damaged)
    out=core.qualify_cycle(x+1e-5,j,1.,1.,4,emit=lambda k,v:events.append((k,v)))
    assert out['status']=='numerical_failure'
    assert events[0][0]=='iterate_return' and events[0][1]['phase_jacobians'].shape==(4,2,2)
    assert any(k=='linear_return' for k,v in events)

def test_prewrite_reservation_leaves_no_partial_npz(tmp_path):
    spec=importlib.util.spec_from_file_location('clark_store',Path(__file__).resolve().parents[1]/'tools/explore_clark_rnn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.ACTIVE_ROOT=tmp_path;m.RAW_CAP=5000;m.FINAL_RESERVE=1000
    with pytest.raises(OSError):m.save_npz(tmp_path/'blocked.npz',x=np.ones(1000))
    assert not (tmp_path/'blocked.npz').exists()
    m.save(tmp_path/'result.json',{'status':'partial'},final=True);m.manifest(tmp_path)
    assert (tmp_path/'manifest.json').exists()

def test_successful_emit_contains_both_native_returns():
    j,x=p4();events=[];out=qualify_cycle(x,j,1.,1.,4,emit=lambda k,v:events.append((k,v)))
    assert out['status']=='numerical_attracting_primitive'
    assert [k for k,v in events]==['iterate_return','adjacent_iterate_return','eigen_return','adjacent_eigen_return']

def test_launcher_cleanup_empty_and_owned_group():
    import subprocess,sys
    spec=importlib.util.spec_from_file_location('clark_launch',Path(__file__).resolve().parents[1]/'tools/launch_clark_rnn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    child=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True);child.wait()
    assert m.cleanup_group(child.pid) and not m.group_alive(child.pid)


def test_components_keep_period_and_boundary_distinct():
    spec=importlib.util.spec_from_file_location('clark_summary',Path(__file__).resolve().parents[1]/'tools/summarize_clark_rnn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=np.array([[0,0,0,0],[0,3,3,0],[0,0,4,0],[0,0,0,0]]);positive=np.zeros_like(p,dtype=bool);positive[0,1]=True
    c=m.components(p,positive);assert [x['period'] for x in c]==[3,4];assert c[0]['size']==2 and not c[0]['touches_grid_edge'];assert c[0]['neighbor_positive_cells']==[(0,1)]
