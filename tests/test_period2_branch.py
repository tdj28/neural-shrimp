"""Synthetic controls only: no development anchor or trajectory is evaluated."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp import period2_branch as b
from neural_shrimp.period2_branch_io import Store,load_record


def family(params):
    center=float(sum(params))
    def f(x):
        z=np.asarray(x)-center
        return center-z+.2*z*(z*z-1)
    def jac(x):
        z=np.asarray(x)-center
        return np.diag(-1+.2*(3*z*z-1))
    return f,jac


def engine(emit=lambda k,v:None):return b.Continuation(family,b.Options(),b.Budget(30,160),emit)
def anchor(e):return e.node(np.array([0.,0.]),[np.array([-1.]),np.array([1.])],'anchor')


def test_translated_cycle_direct_half_reverse_and_two_directions():
    e=engine();root=anchor(e);assert root['accepted']
    for direction in (-1,1):
        segment,last,previous=e.segment(root,np.array([0.,direction*.002]))
        assert segment['status']=='connected'
        assert np.max(np.abs(last['pair']-np.array([[-1.],[1.]])-direction*.002))<1e-12
        edge=e.edges[-1]
        assert all(edge[k]['passed'] for k in ('parent_match','half_parent_match','half_child_match','direct_half_match','reverse_match'))


def test_phase_swap_is_explicit_and_ambiguity_rejected():
    o=b.Options();pair=np.array([[-1.,0.],[1.,0.]])
    match=b.phase_match(pair,pair[::-1],o,replay=True)
    assert match['passed'] and match['phase']==1
    assert not b.phase_match(pair,np.zeros_like(pair),o)['passed']
    assert not b.phase_match(pair,pair+.1,o)['passed']


def test_collapsed_fixed_point_rejected_before_spectrum():
    e=engine();result=e.node([0.,0.],[np.array([0.]),np.array([0.])],'collapse')
    assert not result['accepted'] and result['reason']=='state_gate'
    assert 'spectra' not in result


def test_remote_root_and_iteration_failure_are_not_accepted():
    f,j=family([0,0])
    r=b.correct_phase(f,j,np.array([.8]),b.Options(),b.Budget(),lambda k,v:None)
    assert not r['converged'] and r['reason']=='remote_corrected_phase'
    r=b.correct_phase(f,j,np.array([.999]),b.Options(newton_iterations_per_corrector=1),b.Budget(),lambda k,v:None)
    assert not r['converged'] and r['reason']=='newton_iteration_limit'


def test_nonnormal_complex_left_convention_and_complete_spectrum():
    matrix=.8*np.array([[.6,-4.],[.16,.6]])
    s=b.spectral_checks(np.eye(2),matrix,b.Options())
    assert s['passed']
    assert np.allclose(s['phases'][0]['eigenvalue_condition'],2.6)
    assert np.max(s['phases'][0]['left_residual'])<1e-14
    with pytest.raises(ValueError,match='incomplete'):
        b.eigen_residuals(matrix,np.array([1.]),np.eye(2),np.eye(2))


def test_false_eigenpair_fails_residual_gate(monkeypatch):
    real=b.eig
    def bad(*args,**kwargs):
        values,left,right=real(*args,**kwargs)
        return values+.1,left,right
    monkeypatch.setattr(b,'eig',bad)
    assert not b.spectral_checks(np.eye(2),np.diag([.3,.5]),b.Options())['passed']


@pytest.mark.parametrize('purpose,reason',[('output:half_endpoint','direct_half_mismatch'),('output:reverse','reverse_mismatch')])
def test_direct_half_and_reverse_disagreement_are_rejected(purpose,reason):
    e=engine();root=anchor(e);real=e.node
    def altered(params,seeds,label):
        result=real(params,seeds,label)
        if label==purpose and result['accepted']:result['pair']=result['pair']+1e-5
        return result
    e.node=altered
    edge=e.link(root,np.array([0.,.001]))
    assert edge['status']=='rejected' and edge['reason']==reason


def test_gap_not_jumped_and_rejected_attempts_retained():
    events=[];e=engine(lambda k,v:events.append((k,deepcopy(v))));root=anchor(e)
    base=e.family
    def gap(params):
        if params[1]>=.00045:raise ValueError('synthetic inaccessible branch')
        return base(params)
    e.family=gap
    segment,last,_=e.segment(root,np.array([0.,.001]))
    assert segment['status']=='unresolved_gap' and last['params'][1]<.00045
    assert segment['parent']==root['id']
    assert any(edge['status']=='rejected' for edge in e.edges)
    assert all(e.edges[int(key.split('-')[1])]['status']=='accepted' for key in segment['path'])


def test_bounded_halving_can_resolve_without_accepting_rejected_edge():
    e=engine();root=anchor(e);real=e.link;calls=0
    def first_failure(*args,**kwargs):
        nonlocal calls
        calls+=1
        edge=real(*args,**kwargs)
        if calls==1:edge.update(status='rejected',reason='injected_first_trial_failure')
        return edge
    e.link=first_failure
    segment,_,_=e.segment(root,np.array([0.,.001]))
    assert segment['status']=='connected' and len(segment['path'])==2
    assert e.edges[0]['status']=='rejected'


def test_spectrum_timeout_retains_qualified_pair_pending(monkeypatch):
    events=[];e=engine(lambda k,v:events.append((k,deepcopy(v))))
    def fail(*args,**kwargs):raise b.PilotStop('synthetic spectral deadline')
    monkeypatch.setattr(b,'spectral_checks',fail)
    with pytest.raises(b.PilotStop):anchor(e)
    assert any(k=='pair_spectrum_pending' and v['pair'].shape==(2,1) for k,v in events)
    assert not any(k=='node' for k,v in events)


def test_deadline_and_phase_count_stop_are_retained(tmp_path):
    store=Store(tmp_path/'deadline',65536)
    clock=[0.];budget=b.Budget(1,2,clock=lambda:clock[0])
    def emit(kind,payload):
        store.emit(kind,payload)
        if kind=='corrector_iteration':clock[0]=2
    f,j=family([0,0])
    with pytest.raises(b.PilotStop):b.correct_phase(f,j,np.array([.98]),b.Options(),budget,emit)
    store.finish(dict(status='resource_stop'))
    records=[load_record(store.root,p.name) for p in store.root.glob('*corrector_end.json')]
    assert records and records[0][1]['reason']=='interrupted'
    assert (store.root/'manifest.json').exists()
    budget=b.Budget(10,1);budget.corrector()
    with pytest.raises(b.PilotStop,match='corrector_limit'):budget.corrector()


def test_store_bound_is_hard_and_preserves_failed_attempt(tmp_path):
    store=Store(tmp_path/'full',8192)
    store.emit('small',{'state':np.zeros(2)})
    with pytest.raises(b.PilotStop,match='byte_limit'):store.emit('large',{'state':np.zeros(8192)})
    store.finish(dict(status='resource_stop'))
    assert sum(p.stat().st_size for p in store.root.iterdir())<=8192
    with pytest.raises(FileExistsError):Store(store.root)


def test_bad_controls_rejected():
    for value in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):b.Options(eigen_residual=value)
        with pytest.raises(ValueError):b.Budget(value)


def test_checker_rejects_a_claimed_accepted_edge_through_failed_node():
    spec=importlib.util.spec_from_file_location('s1r_checker',Path(__file__).parents[1]/'tools/check_s1r_branch.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    e=engine();root=anchor(e);edge=e.link(root,np.array([0.,.001]))
    mod.check_edge(edge,e.nodes,b.Options())
    e.nodes[edge['half']]['accepted']=False
    with pytest.raises(ValueError,match='unqualified'):mod.check_edge(edge,e.nodes,b.Options())


def test_complete_cli_and_offline_replay_on_synthetic_72d_family(tmp_path,monkeypatch):
    import sys
    from dataclasses import dataclass
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
    import explore_s1r_branch as runner
    import check_s1r_branch as checker
    @dataclass
    class SyntheticModel:
        n_parameters:int=72
    def fake_system(seed):
        assert seed==28002
        return SyntheticModel(),np.zeros((1,1)),np.zeros(1),np.zeros(72)
    def fake_family(params):
        center=float(sum(params))
        def f(x):
            out=.5*np.asarray(x);z=x[0]-center
            out[0]=center-z+.2*z*(z*z-1)
            return out
        def jac(x):
            out=.5*np.eye(72);z=x[0]-center
            out[0,0]=-1+.2*(3*z*z-1)
            return out
        return f,jac
    class SyntheticGrid:
        def __init__(self,net,x,y,hidden,output):self.params=np.log10([hidden[0],output[0]])
        def cell_map(self,index):return fake_family(self.params)
    def fake_seed(plan):
        seed=np.zeros(72);seed[0]=plan['log10_hidden_anchor']+plan['log10_output_anchor']-1
        return seed
    monkeypatch.setitem(runner.SYSTEMS,'S1R',fake_system)
    monkeypatch.setattr(runner,'bound_inputs',fake_seed)
    monkeypatch.setattr(runner,'GridMap',SyntheticGrid)
    monkeypatch.setattr(checker,'GridMap',SyntheticGrid)
    real_run=runner.subprocess.run
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:real_run(*a,**kw) if a[0][1]=='rev-parse' else None)
    root=tmp_path/'synthetic-cli'
    result=runner.run(root)
    assert result['status']=='completed_local_segment' and result['phase_corrector_calls']==42
    replay=checker.check(root)
    assert replay['attempt_status']=='completed_local_segment' and replay['accepted_edges']==4

    # A self-consistently rehashed false graph claim must still fail semantically.
    import json,shutil
    from neural_shrimp.period2_branch_io import file_entry
    tampered=tmp_path/'false-graph';shutil.copytree(root,tampered)
    tampered.chmod(0o755)
    for path in tampered.iterdir():path.chmod(0o644)
    outcome=json.loads((tampered/'result.json').read_text())
    outcome['segments'][0]['path']=['edge-9999']
    (tampered/'result.json').write_text(json.dumps(outcome))
    manifest=json.loads((tampered/'manifest.json').read_text())
    manifest['files']['result.json']=file_entry(tampered/'result.json')
    (tampered/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='segment graph differs'):checker.check(tampered)
