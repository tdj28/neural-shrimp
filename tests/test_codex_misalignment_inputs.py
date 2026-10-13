import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SRC=Path(__file__).resolve().parents[1]/'codex/misalignment-audit'
sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location('audit_inputs',SRC/'inputs.py')
I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)
import detection as D


def fake():
    return {'q':np.zeros((2,40)), 'q_plus':np.ones((2,40)),
            'pre':np.zeros((2,40,24)), 'axis':np.zeros((2,40)),
            'score':np.zeros((2,40)), 'tok':np.zeros((2,40,3)), 'dose':.04}


def test_l7_unused_terminal_nan_recorded_without_feature_change():
    arm=fake(); arm['q'][:,39]=np.nan; arm['q_plus'][1,39]=np.nan
    I.validate_detection_finiteness(arm,'all')
    assert arm['feature_admission']['probes']['q']=={
        'nan_total':2,'selected_probe_entries':78,'nan_unused':2}
    for h in (1,3,5):
        for run in (1,2):
            table=D.make_table([arm],h,run,'all')
            assert table['turn'].max()<=38
            np.testing.assert_array_equal(table['extra'],np.tile([0.,1.,0.],(len(table['y']),1)))


@pytest.mark.parametrize('key,turn,value',[('q',38,np.nan),('q_plus',0,np.nan),('q',39,np.inf)])
def test_bad_used_or_infinite_probe_rejected(key,turn,value):
    arm=fake(); arm[key][0,turn]=value
    with pytest.raises(ValueError,match='probe'): I.validate_detection_finiteness(arm,'all')


def test_all_nan_probes_allowed_only_on_baseline_state_path():
    arm=fake(); arm['q'][:]=np.nan; arm['q_plus'][:]=np.nan
    with pytest.raises(ValueError,match='used L7 probe'): I.validate_detection_finiteness(arm,'all')
    I.validate_detection_finiteness(arm,'baseline_state')
    assert arm['feature_admission']['probes']['q']['nan_unused']==80
    assert arm['feature_admission']['unused_probes_imputed'] is False
    # Removing probe/token keys proves table construction does not read them.
    del arm['q'],arm['q_plus'],arm['tok']
    tab=D.make_table([arm],1,1,'baseline_state')
    assert tab['extra'].shape==(78,0)
    assert D.predictor_features(tab,np.zeros((78,20)),'D3_state_probe').shape==(78,26)
    for predictor in ('D1_forecast','D2_susceptibility','D4_token_max','all'):
        with pytest.raises(ValueError,match='probe features absent'):
            D.predictor_features(tab,np.zeros((78,20)),predictor)
    with pytest.raises(ValueError,match='L7 requires'): D.crossfit(tab,[])
    with pytest.raises(ValueError,match='not admitted'): D.make_table([arm],1,1,'all')


@pytest.mark.parametrize('feature_set',['all','baseline_state'])
@pytest.mark.parametrize('key',['score','axis','pre'])
def test_nonfinite_baseline_or_state_rejected(feature_set,key):
    arm=fake(); arm[key].flat[0]=np.nan
    with pytest.raises(ValueError,match='baseline/state'):
        I.validate_detection_finiteness(arm,feature_set)


def test_actual_table_still_rejects_nonfinite_used_features():
    arm=fake(); arm['q'][0,1]=np.nan
    with pytest.raises(ValueError,match='nonfinite feature'): D.make_table([arm],1,1)
    arm=fake(); arm['pre'][0,1,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite feature'): D.make_table([arm],1,1,'baseline_state')


def test_unused_post_event_probes_are_not_imputed_or_selected():
    arm=fake(); arm['score'][0,2:4]=.8
    arm['q'][0,2:]=np.nan; arm['q_plus'][0,2:]=np.nan
    I.validate_detection_finiteness(arm,'all')
    for event in (1,2):
        tab=D.make_table([arm],1,event)
        assert np.all(tab['turn'][tab['group']==0]<2)
        assert np.isfinite(tab['extra']).all()


def test_frozen_training_and_test_table_values_unchanged():
    arm=fake(); arm['axis']=np.arange(80).reshape(2,40); arm['pre'][:]=2
    all_features=D.make_table([arm],3,2,'all')
    frozen=D.make_table([arm],3,2,'baseline_state')
    for key in ('y','group','base','state','arm','turn'):
        np.testing.assert_array_equal(frozen[key],all_features[key])


def test_repaired_run_requires_exact_closed_failed_inventory(tmp_path,monkeypatch):
    import json
    spec=importlib.util.spec_from_file_location('audit_run',SRC/'run.py')
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    monkeypatch.setattr(run,'ALLOWED',tmp_path)
    roots=[tmp_path/'l7-001',tmp_path/'l7-001-owner']
    for root in roots: root.mkdir()
    (roots[0]/'result.json').write_text(json.dumps({'completed':False}))
    (roots[1]/'result.json').write_text(json.dumps({'owned_worker_exited':True,'worker_exit':2}))
    card={'prior_failure_inventory':[{'path':str(p),'bytes':p.stat().st_size,'sha256':run.sha(p)}
           for root in roots for p in sorted(root.rglob('*')) if p.is_file()]}
    run.verify_prior_failure(card)
    (roots[0]/'unexpected').write_text('retained additional history')
    with pytest.raises(ValueError,match='inventory differs'):run.verify_prior_failure(card)


def test_failed_inventory_cannot_claim_an_unclosed_worker(tmp_path,monkeypatch):
    import json
    spec=importlib.util.spec_from_file_location('audit_run2',SRC/'run.py')
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    monkeypatch.setattr(run,'ALLOWED',tmp_path)
    roots=[tmp_path/'l7-001',tmp_path/'l7-001-owner']
    for root in roots: root.mkdir()
    (roots[0]/'result.json').write_text(json.dumps({'completed':False}))
    (roots[1]/'result.json').write_text(json.dumps({'owned_worker_exited':False,'worker_exit':2}))
    card={'prior_failure_inventory':[{'path':str(p),'bytes':p.stat().st_size,'sha256':run.sha(p)}
           for root in roots for p in sorted(root.rglob('*')) if p.is_file()]}
    with pytest.raises(ValueError,match='termination status'):run.verify_prior_failure(card)
