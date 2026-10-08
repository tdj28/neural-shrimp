"""Independent synthetic source controls; never evaluates a saved target gradient."""
from dataclasses import dataclass,field
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np
import pytest
from neural_shrimp.period2_branch_io import load_record
from neural_shrimp.utility_dose import dose_prediction

ROOT=Path(__file__).resolve().parents[1]
def runner():
    spec=importlib.util.spec_from_file_location('independent_dose_runner',ROOT/'tools/explore_utility_dose_feasibility.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def records(out):
    lines=[json.loads(line) for line in (out/'journal.jsonl').read_text().splitlines()]
    return [load_record(out,line['file']) for line in lines]

def setup_run(monkeypatch,mode=None):
    obj=runner()
    @dataclass
    class SyntheticNet:
        n_parameters:int=13
        hidden_parameter_count:int=9
        observed:list=field(default_factory=list)
        def gradient(self,state,x,y):
            index=len(self.observed);self.observed.append(state.copy())
            np.testing.assert_array_equal(state[0],np.arange(13,dtype=float)+index)
            if mode=='fatal' and index==1:raise RuntimeError('synthetic gradient failure')
            if mode=='term' and index==0:signal.getsignal(signal.SIGTERM)(None,None)
            if mode=='mutation' and index==0:state[0,0]+=1
            value=np.linspace(-1.,2.,13)
            if mode=='nan':value[0]=np.nan
            return value[None,:]
        def loss(self,*args,**kwargs):raise AssertionError('loss not authorized')
        def hvp(self,*args,**kwargs):raise AssertionError('HVP not authorized')
        def predict(self,*args,**kwargs):raise AssertionError('extra prediction not authorized')
    net=SyntheticNet()
    forks=[dict(point=i//2,origin=str(i%2),native_state=np.arange(13,dtype=float)+i,native_rates=np.array([2.,.5]),state={'sha256':'synthetic'}) for i in range(4)]
    monkeypatch.setattr(obj,'sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(obj,'bound_inputs',lambda:(net,np.ones((3,2)),np.zeros(3),forks,{'projected_seconds':1.2}))
    monkeypatch.setattr(obj.os,'getpriority',lambda *_:10)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return obj,net,forks,head

def test_independent_block_dose_and_total_step_oracle():
    g=np.linspace(-1.,2.,13);rates=np.array([2.,.5]);s=g*np.r_[np.repeat(2.,9),np.repeat(.5,4)]
    result=dose_prediction(g,rates,9);R=.05*np.linalg.norm(s)
    expected={
        'sham':np.zeros(13),'uniform':.05*s,
        'hidden':np.r_[R*s[:9]/np.linalg.norm(s[:9]),np.zeros(4)],
        'output':np.r_[np.zeros(9),R*s[9:]/np.linalg.norm(s[9:])]}
    for name,delta in expected.items():
        arm=result['arm_predictions'][name]
        np.testing.assert_allclose(arm['treatment_minus_sham_increment'],delta,rtol=1e-14,atol=1e-15)
        np.testing.assert_allclose(arm['predicted_update_increment'],-s+delta,rtol=1e-14,atol=1e-15)
        assert arm['predicted_total_step_length']==pytest.approx(np.linalg.norm(-s+delta),rel=1e-14)
        assert arm['dose']==pytest.approx(0 if name=='sham' else R,abs=1e-14)
    assert result['hidden_slice']==[0,9] and result['output_slice']==[9,13]
    assert np.count_nonzero(expected['hidden'][6:9])==3 and expected['output'][12]!=0
    assert len({v['predicted_total_step_length'] for v in result['arm_predictions'].values()})>1

@pytest.mark.parametrize('mode',['nan','fatal','term','mutation'])
def test_all_bound_forks_and_returned_gradients_survive_failures(tmp_path,monkeypatch,mode):
    obj,net,forks,head=setup_run(monkeypatch,mode);out=tmp_path/mode
    if mode in ('fatal','mutation'):
        with pytest.raises((ValueError,RuntimeError)):obj.run(out,head)
        result=json.loads((out/'result.json').read_text())
    else:result=obj.run(out,head)
    stages=records(out);bound=next(v for k,v in stages if k=='bound_inputs')
    assert len(bound['forks'])==4
    for i,f in enumerate(bound['forks']):np.testing.assert_array_equal(f['state'],np.arange(13)+i)
    returned=[v for k,v in stages if k=='gradient_return']
    assert len(returned)==result['gradient_returns']
    if mode=='nan':
        assert net.observed.__len__()==result['gradient_attempts']==4 and len(returned)==4
        assert result['status']=='fixed_candidate_infeasible' and not result['candidate_feasible']
        assert all(np.isnan(v['gradient'][0]) for v in returned)
    elif mode=='fatal':assert result['gradient_attempts']==2 and result['gradient_returns']==1 and result['status']=='failed_or_interrupted'
    else:
        assert result['gradient_attempts']==result['gradient_returns']==1
        assert not any(k=='dose_prediction' for k,v in stages)
        assert result['status']==('resource_stop' if mode=='term' else 'failed_or_interrupted')
    assert result['gradient_attempts']<=4 and sum(p.stat().st_size for p in out.iterdir())<768*1024
    manifest=json.loads((out/'manifest.json').read_text())
    assert set(manifest['files'])=={p.name for p in out.iterdir()}-{'manifest.json'}
    assert all(not (p.stat().st_mode&0o222) for p in out.iterdir())
    for i,f in enumerate(forks):np.testing.assert_array_equal(f['native_state'],np.arange(13)+i)

def test_source_inventory_includes_package_transitive_imports():
    obj=runner()
    assert {'src/neural_shrimp/controls.py','src/neural_shrimp/models.py'}<=set(obj.SOURCES)

def test_dirty_imported_source_is_rejected(tmp_path,monkeypatch):
    obj=runner();monkeypatch.setattr(obj,'ROOT',tmp_path)
    for name in obj.SOURCES:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'committed')
    def git_output(args,**kwargs):
        if args[1]=='status':return ' M src/neural_shrimp/models.py\n' if kwargs.get('text') else b' M src/neural_shrimp/models.py\n'
        if args[1]=='show':return b'committed'
        if args[1]=='rev-parse':return 'a'*40 if kwargs.get('text') else b'a'*40
        raise AssertionError(args)
    monkeypatch.setattr(obj.subprocess,'check_output',git_output)
    with pytest.raises(ValueError):obj.sources()

def test_deadline_after_final_prediction_cannot_report_success(tmp_path,monkeypatch):
    obj,net,forks,head=setup_run(monkeypatch);clock=[0.]
    monkeypatch.setattr(obj.time,'monotonic',lambda:clock[0])
    base=obj.Store
    class LateFinalWrite(base):
        def emit(self,kind,payload,**kw):
            value=super().emit(kind,payload,**kw)
            if kind=='dose_prediction' and payload['index']==3:clock[0]=6.
            return value
    monkeypatch.setattr(obj,'Store',LateFinalWrite)
    result=obj.run(tmp_path/'late-final',head)
    assert result['gradient_returns']==4
    assert result['status']=='resource_stop' and result['candidate_feasible'] is None
