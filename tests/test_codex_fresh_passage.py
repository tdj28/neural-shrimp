"""Task5 frozen-design/passages tests using CPU float64 algebraic maps only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / 'codex/confirm-fold'
if str(CODE) not in sys.path:
    sys.path.insert(0,str(CODE))
import build_design as design
import passage


def write_setup(tmp_path, label, *, raw_a=12.,raw_b=10.,passed=True):
    folder=tmp_path / ('setup-'+label)
    folder.mkdir()
    x=np.array([.2,-.3,.4])
    raw_r=np.array([1.,0.,0.]);raw_l=np.array([1.,4.,-2.])
    orientation=np.sign(raw_b)
    r,l=orientation*raw_r,orientation*raw_l
    matrix=.2*np.eye(3)+.8*np.outer(r,l)
    np.savez(folder/'setup.npz',x_c=x,r=r,l=l,matrix=matrix)
    info=dict(target=dict(label=label,period=2 if label=='A' else 8,
                          coordinate='alpha' if label=='A' else 'log10T'),passed=passed,
              array_file_sha256=design.sha(folder/'setup.npz'),parameter_c=.067 if label=='A' else -.1033,
              alpha=.067 if label=='A' else .0717,log10T=-.140 if label=='A' else -.1033,
              raw_a=raw_a,raw_b=raw_b,a_positive=abs(raw_a),b_positive=abs(raw_b),
              ghost_parameter_direction=int(np.sign(raw_a*raw_b)))
    (folder/'setup.json').write_text(json.dumps(info))
    return folder/'setup.json'


def target_fixture(tmp_path,label='A',**kwargs):
    path=write_setup(tmp_path,label,**kwargs)
    inputs=tmp_path/'inputs';inputs.mkdir()
    target=design.build_target(label,path,inputs)
    arrays=np.load(target['inputs']['path'],allow_pickle=False) if target['status']=='ready' else None
    return target,arrays


class QuadraticReturn:
    """Exact quadratic G=F^p surrogate with a nonnormal transverse projection."""
    def __init__(self,target,arrays):
        self.t=target
        self.x=torch.tensor(arrays['x_c'],dtype=torch.float64)
        self.r=torch.tensor(arrays['r'],dtype=torch.float64)
        self.l=torch.tensor(arrays['l'],dtype=torch.float64)
        self.calls=[]
    def prefix_cache(self,alpha):
        return alpha
    def period_map(self,states,alpha,logT,period,cache=None):
        assert period==self.t['period']
        assert states.dtype==torch.float64 and states.device.type=='cpu'
        p=alpha if self.t['coordinate']=='alpha' else logT
        eps=(p-self.t['parameter_c'])*self.t['ghost_direction']
        assert eps>0
        self.calls.append((alpha,logT,period,len(states)))
        diff=states-self.x
        u=diff@self.l
        transverse=diff-u[:,None]*self.r
        up=u+self.t['a_positive']*eps+self.t['b_positive']*u*u
        return self.x+up[:,None]*self.r+.2*transverse


@pytest.mark.parametrize('label',['A','B'])
@pytest.mark.parametrize('raw_a,raw_b',[(12.,10.),(12.,-10.),(-12.,10.),(-12.,-10.)])
def test_generated_design_and_exact_quadratic_passages_p2_p8_signs_and_clock(tmp_path,label,raw_a,raw_b):
    target,arrays=target_fixture(tmp_path,label,raw_a=raw_a,raw_b=raw_b)
    assert target['period']==(2 if label=='A' else 8)
    assert target['ghost_direction']==np.sign(raw_a*raw_b)
    assert target['geometry']['transverse_kick_norm']==pytest.approx(target['U'])
    assert abs(target['geometry']['transverse_projection'])<1e-13
    assert target['geometry']['initial_u']==pytest.approx([-target['U'],0.,-target['U']],abs=1e-14)
    model=QuadraticReturn(target,arrays)
    saves=[]
    rows,meta=passage.run_group(model,target,0,arrays,lambda value:saves.append(copy.deepcopy(value)))
    assert [r['entrance'] for r in rows]==['full','midpoint','full+transverse']
    assert all(r['status']=='observed' and r['primary_pass'] for r in rows)
    assert all(r['measured_G']==r['quadratic_G'] for r in rows)
    assert all(r['measured_F']==target['period']*r['measured_G'] for r in rows)
    assert meta['G_evaluations']==max(r['measured_G'] for r in rows)
    assert meta['batch_model_steps']==target['period']*meta['G_evaluations']
    assert len(model.calls)==meta['G_evaluations']
    physical=target['parameter_c']+target['ghost_direction']*design.EPSILONS[0]
    assert all(call[0 if label=='A' else 1]==physical for call in model.calls)
    assert saves
    final=saves[-1]
    assert final['state_trace'].shape==(meta['G_evaluations'],3,3)
    for key in ('initial_states','previous_states','crossing_states','states_at_stop'):
        assert final[key].shape==(3,3)
    for j,row in enumerate(rows):
        projected=final['u'][:row['measured_G'],j]
        assert np.isfinite(projected).all()
        assert projected[-1]>target['U']
        assert ((final['previous_states'][j]-arrays['x_c'])@arrays['l'])<=target['U']+1e-14
        assert ((final['crossing_states'][j]-arrays['x_c'])@arrays['l'])>target['U']
        assert np.isnan(final['u'][row['measured_G']:,j]).all()
        retained=final['state_trace'][:row['measured_G'],j]
        assert np.isfinite(retained).all()
        assert (retained-arrays['x_c'])@arrays['l']==pytest.approx(projected)
        assert np.array_equal(retained[-1],final['crossing_states'][j])
        assert np.isnan(final['state_trace'][row['measured_G']:,j]).all()
        assert np.array_equal(final['states_at_stop'][j],final['crossing_states'][j])


def test_transverse_means_kernel_of_left_mode_not_euclidean_orthogonal_to_right():
    x=np.array([.2,.3,.4]);r=np.array([1.,0.,0.]);l=np.array([1.,5.,-3.])
    w,starts=design.entrances(x,r,l,.1,10501)
    assert abs(l@w)<1e-13
    assert np.linalg.norm(w)==pytest.approx(1.)
    assert abs(r@w)>.1  # Oblique center projection is intentional.
    assert (starts-x)@l==pytest.approx([-.1,0.,-.1])
    assert np.linalg.norm(starts[2]-starts[0])==pytest.approx(.1)
    w2,starts2=design.entrances(x,r,l,.1,10501)
    assert np.array_equal(w,w2) and np.array_equal(starts,starts2)


def small_target(period=2,max_g=10):
    rows=[]
    for j,name in enumerate(design.ENTRANCES):
        rows.append(dict(id=str(j),group=0,entrance_index=j,entrance=name,parameter=.1,
                         quadratic_G=3 if j!=1 else 2,arctan_G=3. if j!=1 else 2.))
    t=dict(rows=rows,period=period,max_g=max_g,coordinate='alpha',alpha=.1,log10T=0.,U=.5)
    arrays=dict(x_c=np.zeros(2),r=np.array([1.,0.]),l=np.array([1.,0.]),
                starts=np.array([[-.5,0.],[0.,0.],[-.5,.5]]))
    return t,arrays


class Increment:
    def prefix_cache(self,alpha):
        return alpha
    def period_map(self,states,alpha,logT,period,cache=None):
        return states+torch.tensor([.5,0.],dtype=torch.float64)


def test_strict_exit_midpoint_and_no_burn_in_count():
    t,arrays=small_target(period=8)
    saves=[]
    rows,meta=passage.run_group(Increment(),t,0,arrays,saves.append)
    # Full: -.5 -> 0 -> .5 (not exit) -> 1. Midpoint: 0 -> .5 -> 1.
    assert [r['measured_G'] for r in rows]==[3,2,3]
    assert [r['measured_F'] for r in rows]==[24,16,24]
    assert meta['batch_model_steps']==24
    assert saves[-1]['u'][0,1]==.5
    assert rows[1]['primary_pass']


def test_cap_censors_each_unfinished_row_and_preserves_states():
    t,arrays=small_target(max_g=1)
    saves=[]
    rows,meta=passage.run_group(Increment(),t,0,arrays,saves.append)
    assert all(r['status']=='censored' and r['measured_G'] is None for r in rows)
    assert all(not r['primary_pass'] and not r['arctan_pass'] for r in rows)
    assert all(r['recorded_G']==1 for r in rows)
    assert all(r['measured_F'] is None for r in rows)
    assert np.isnan(saves[-1]['crossing_states']).all()
    assert np.array_equal(saves[-1]['states_at_stop'],arrays['starts']+[.5,0.])
    assert meta['G_evaluations']==1


def test_generated_failed_target_retains_all_not_run_rows(tmp_path):
    target,_=target_fixture(tmp_path,passed=False)
    assert target['status']=='setup_gate_failed'
    assert len(target['rows'])==12
    assert {r['status'] for r in target['rows']}=={'not_run_setup_gate'}


def test_setup_arrays_mutation_rejected_before_design(tmp_path):
    source=write_setup(tmp_path,'A')
    (source.parent/'setup.npz').write_bytes(b'changed')
    inputs=tmp_path/'inputs';inputs.mkdir()
    with pytest.raises(ValueError,match='hash mismatch'):
        design.build_target('A',source,inputs)


def plan_fixture(tmp_path,monkeypatch):
    target,arrays=target_fixture(tmp_path)
    repo=tmp_path/'repo';repo.mkdir()
    source=repo/'worker.py';source.write_text('ORIGINAL = True\n')
    monkeypatch.setattr(passage,'REPO',repo)
    plan=dict(schema=1,execution=dict(threads=4,device='cpu',dtype='float64'),
              source=[dict(path='worker.py',sha256=design.sha(source))],software={},targets=[target])
    path=tmp_path/'design.json';path.write_text(json.dumps(plan))
    return path,plan,target,source


def test_frozen_plan_source_mutation_rejected(tmp_path,monkeypatch):
    path,_,_,source=plan_fixture(tmp_path,monkeypatch)
    passage.validate_plan(path,'A',4)
    source.write_text('MUTATED = True\n')
    with pytest.raises(ValueError,match='source changed'):
        passage.validate_plan(path,'A',4)


@pytest.mark.parametrize('which',['setup','setup_arrays','inputs'])
def test_frozen_plan_artifact_mutation_rejected(tmp_path,monkeypatch,which):
    path,_,target,_=plan_fixture(tmp_path,monkeypatch)
    artifact=Path(target[which]['path'])
    artifact.chmod(0o644)
    artifact.write_bytes(artifact.read_bytes()+b' mutation')
    with pytest.raises(ValueError,match='bound input changed'):
        passage.validate_plan(path,'A',4)


def test_frozen_plan_thread_or_target_identity_mismatch_rejected(tmp_path,monkeypatch):
    path,plan,_,_=plan_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='execution contract'):
        passage.validate_plan(path,'A',2)
    with pytest.raises(ValueError,match='unknown or repeated target'):
        passage.validate_plan(path,'B',4)
    plan['targets'].append(plan['targets'][0])
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='unknown or repeated target'):
        passage.validate_plan(path,'A',4)


def test_malformed_entrance_group_rejected():
    t,arrays=small_target()
    t['rows'][2]['entrance_index']=1
    with pytest.raises(ValueError,match='three-entrance'):
        passage.run_group(Increment(),t,0,arrays,lambda _:None)


class ForbiddenPostExit(Increment):
    def period_map(self,states,alpha,logT,period,cache=None):
        # Domain ends immediately beyond u=.5; no stopped column may revisit it.
        if bool(torch.any(states[:,0]>.5)):
            raise RuntimeError('post-exit state was evaluated')
        return super().period_map(states,alpha,logT,period,cache)


def test_inactive_columns_never_evolve_from_crossed_states():
    t,arrays=small_target()
    saved=[]
    rows,_=passage.run_group(ForbiddenPostExit(),t,0,arrays,saved.append)
    assert [row['measured_G'] for row in rows]==[3,2,3]
    assert np.array_equal(saved[-1]['states_at_stop'],saved[-1]['crossing_states'])


def test_every_generated_prefix_survives_interruption():
    t,arrays=small_target()
    saved=[];calls=0
    def tick():
        nonlocal calls
        calls+=1
        if calls==3:
            raise TimeoutError('synthetic interruption')
    with pytest.raises(TimeoutError,match='synthetic interruption'):
        passage.run_group(Increment(),t,0,arrays,saved.append,tick=tick)
    assert saved[-1]['state_trace'].shape==(2,3,2)
    assert np.array_equal(saved[-1]['state_trace'][-1],arrays['starts']+[1.,0.])
    assert saved[-1]['u'][:,1].tolist()==[.5,1.]
    assert np.isfinite(saved[-1]['crossing_states'][1]).all()
    assert np.isnan(saved[-1]['crossing_states'][0]).all()


@pytest.mark.parametrize('failure',['source_commit','pushed_at','plan_hash'])
def test_launcher_provenance_or_plan_mutation_rejected_before_model(tmp_path,monkeypatch,failure):
    repo=tmp_path/'snapshot';folder=repo/'codex/confirm-fold';folder.mkdir(parents=True)
    path=folder/'design.json';path.write_text('{}')
    monkeypatch.setattr(passage,'REPO',repo)
    revision='a'*40;pushed='2026-10-13T01:00:00+00:00'
    monkeypatch.setenv('TASK5_SOURCE_COMMIT',revision)
    monkeypatch.setenv('TASK5_PUSH_VERIFIED_AT',pushed)
    monkeypatch.setenv('TASK5_PLAN_SHA256',design.sha(path))
    if failure=='source_commit':
        monkeypatch.setenv('TASK5_SOURCE_COMMIT','b'*40)
    elif failure=='pushed_at':
        monkeypatch.setenv('TASK5_PUSH_VERIFIED_AT','2026-10-13T01:00:01+00:00')
    else:
        path.write_text('{"mutated":true}')
    def no_model(*args,**kwargs):
        raise AssertionError('provenance failure must precede model loading')
    monkeypatch.setattr(passage.ManualQwenSoftMap,'from_pinned_files',no_model)
    monkeypatch.setattr(sys,'argv',['passage.py','--design',str(path),'--target','A','--threads','4',
                                   '--freeze',revision,'--pushed-at',pushed,'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='provenance mismatch|pushed source snapshot'):
        passage.main()
    assert not (tmp_path/'out').exists()
