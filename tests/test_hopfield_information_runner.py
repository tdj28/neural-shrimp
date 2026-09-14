"""Outcome-free contracts for EXP008 chronological seals and full censuses."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_hopfield_information as driver
from hopfield_information_features import base_features, BASE_FEATURE_NAMES


def plan():
    return json.loads(driver.PLAN.read_text())


def development_fixture(p):
    rows = []
    for a in driver.anchors_for(p, 'development'):
        for ph in p['phases']:
            for r in p['radii']:
                for c in range(18):
                    rows.append(dict(id=driver.feature_id(a['id'], ph, r, c), role='development',
                        anchor=a['id'], block=a['block_id'], phase=ph, radius=r, candidate=c,
                        status='complete', base=[1.]*51, geometry=[float(c)]*6))
    records = [{**s, 'id':driver.trial_id(s), 'status':'complete', 'rmse':.1}
               for s in driver.specs(p, 'development')]
    return rows, records


def test_independent_exhaustive_census():
    p = plan()
    rows = [s for role in ['development', 'test'] for s in driver.specs(p, role)]
    assert len(rows) == len({driver.trial_id(s) for s in rows}) == 7008
    assert sum(r['radius']==0 for r in rows) == 96
    assert sum(r['role']=='test' and r['radius']>0 for r in rows) == 3456
    assert sum(len(driver.expected_features(p, role)) for role in ['development','test']) == 3456
    assert p['phases']==[675,1350]
    assert len(BASE_FEATURE_NAMES)==51 and len(set(BASE_FEATURE_NAMES))==51


def test_candidates_keep_exact_axes_and_duplicates():
    ds = driver.candidates([0.,1.])
    np.testing.assert_array_equal(ds[[0,4,8,12]], [[1,0],[0,1],[-1,0],[0,-1]])
    assert ds.shape==(18,2)
    np.testing.assert_array_equal(ds[4],ds[16])
    np.testing.assert_array_equal(ds[12],ds[17])


def test_shuffle_has_fixed_donor_and_matching_context():
    p = plan(); rows,_ = development_fixture(p)
    for row in rows:
        row['geometry']=[int(row['anchor'].split('-')[-1]), row['phase'], row['radius'], row['candidate']]
    actual = driver.shuffled_geometry(rows,p,'development')
    for row, value in zip(rows,actual):
        assert value==[(int(row['anchor'].split('-')[-1])+5)%12,row['phase'],row['radius'],row['candidate']]


@pytest.mark.parametrize('failure', ['coarse','zero','fine','boolean'])
def test_every_required_development_record_gates_fit(monkeypatch, failure):
    p=plan(); rows,records=development_fixture(p)
    predicate = {'coarse':lambda r:r['candidate']>=0 and r['dt']==.01,
                 'zero':lambda r:r['candidate']==-1,
                 'fine':lambda r:r['candidate']>=0 and r['dt']==.005,
                 'boolean':lambda r:True}[failure]
    bad=next(r for r in records if predicate(r))
    if failure=='boolean':bad['rmse']=True
    else:bad['status']='failed';bad['rmse']=None
    captured={}
    def fakefit(x,g,y,*args,**kwargs):
        captured['labels']=y;return {'status':'training_failed'}
    monkeypatch.setattr(driver,'fit_information_models',fakefit)
    assert driver.fit_development(rows,records,p)['status']=='training_failed'
    assert len(captured['labels'])==1728 and all(v is None for v in captured['labels'])


def test_fit_rejects_test_rows_and_missing_development(monkeypatch):
    p=plan();rows,records=development_fixture(p)
    rows[0]['role']='test'
    with pytest.raises(ValueError,match='Only development'):
        driver.fit_development(rows,records,p)
    rows[0]['role']='development'
    with pytest.raises(ValueError,match='Incomplete development target'):
        driver.fit_development(rows,records[:-1],p)


def seals(tmp_path):
    def write(name,value):driver.base.write(tmp_path/name,value)
    write('model.json',{'model':'synthetic'})
    write('model-seal.json',{'model_sha256':driver.base.sha(tmp_path/'model.json'),'sealed_at':'2026-09-13T00:00:01Z'})
    write('test-features.json',[]);write('test-decisions.json',[])
    write('test-decision-seal.json',dict(model_seal_sha256=driver.base.sha(tmp_path/'model-seal.json'),
        features_sha256=driver.base.sha(tmp_path/'test-features.json'),decisions_sha256=driver.base.sha(tmp_path/'test-decisions.json'),
        sealed_at='2026-09-13T00:00:02Z'))


def test_test_targets_require_seals_and_exclusive_journal(tmp_path):
    with pytest.raises(FileNotFoundError):driver.authorize_test_targets(tmp_path)
    seals(tmp_path)
    assert driver.authorize_test_targets(tmp_path)['model_seal_sha256']==driver.base.sha(tmp_path/'model-seal.json')
    (tmp_path/'test-targets.jsonl').write_text('')
    with pytest.raises(AssertionError,match='consumed'):driver.authorize_test_targets(tmp_path)


@pytest.mark.parametrize('file',['model.json','model-seal.json','test-features.json','test-decisions.json'])
def test_seals_reject_mutation(tmp_path,file):
    seals(tmp_path);(tmp_path/file).write_text('{}')
    with pytest.raises((AssertionError,KeyError)):driver.authorize_test_targets(tmp_path)


def test_model_replay_tolerance_does_not_relax_discrete_choices():
    driver.compare_model_replay({'coef':[.1],'chosen_lambda_index':2},{'coef':[.1+1e-10],'chosen_lambda_index':2})
    with pytest.raises(AssertionError):
        driver.compare_model_replay({'coef':[.1],'chosen_lambda_index':2},{'coef':[.1],'chosen_lambda_index':3})
    with pytest.raises(AssertionError):driver.compare_model_replay([.1],[.11])


@pytest.mark.parametrize('role', ['development', 'test'])
def test_failed_coarse_zero_sham_invalidates_global_primary(monkeypatch, role):
    p=plan()
    records=[{**s,'id':driver.trial_id(s),'status':'complete','rmse':.1}
             for group in driver.ROLES for s in driver.specs(p,group)]
    monkeypatch.setattr(driver,'summarize_information_outcomes',lambda *args,**kwargs:
                        {'primary':{'status':'complete'},'steps':{'diagnostic':'retained'}})
    complete=driver.summarize(records,[],p)
    assert complete['primary']['status']=='complete' and complete['primary_data_gate']['passed']
    bad=next(r for r in records if r['role']==role and r['dt']==.01 and r['candidate']==-1)
    bad.update(status='failed',rmse=None)
    result=driver.summarize(records,[],p)
    assert result['primary']['status']=='undefined_incomplete'
    assert not result['primary_data_gate']['passed']
    assert result['steps']['diagnostic']=='retained'
    assert result['zero'][f'{role}-0.01']['mean'] is None


def synthetic_feature_inputs():
    phase=np.arange(2701)/2700
    orbit=np.column_stack([np.sin(2*np.pi*phase+s) for s in [0,.3,.7]])
    cert=dict(period=40.,transverse_spectral_radius=.2,orbit_activations=orbit)
    gram=dict(gram=np.eye(2),unprojected_gram=2*np.eye(2),phase_coefficients=[.2,.3])
    rms={'aggregate':.2,'per_component':[.1,.2,.3]}
    metrics=dict(aligned_rms=rms,synchronous_rms=rms,phase_offset_cycles=.1,
                 windows=[{'aligned_rms':rms},{'aligned_rms':rms}])
    return cert,gram,metrics


def test_features_are_explicit_finite_and_share_crop_knowledge():
    cert,gram,metrics=synthetic_feature_inputs()
    values=base_features({'alpha':1.079,'gamma':10.3},675,[1.,0.],.016,cert,gram,metrics)
    assert values.shape==(51,) and np.isfinite(values).all()
    lookup=dict(zip(BASE_FEATURE_NAMES,values))
    assert lookup['rectangle_contains_destination']==0
    assert lookup['signed_crop_clearance']<0
    assert lookup['local_projected_rmse']==pytest.approx(.016)
    assert lookup['destination_alpha']==pytest.approx(1.095)
    assert lookup['preview_rmse']==.2


def test_symmetric_half_turn_phase_is_no_longer_planned():
    cert,gram,metrics=synthetic_feature_inputs()
    with pytest.raises(ValueError,match='Unplanned phase'):
        base_features({'alpha':1.064,'gamma':10.26},2025,[1.,0.],.001,cert,gram,metrics)


def test_mocked_full_driver_orders_model_and_decisions_before_test_targets(tmp_path,monkeypatch):
    """No ODE executes: exercise actual run orchestration/write/receipt boundary."""
    p=plan();events=[]
    monkeypatch.setattr(driver,'validate',lambda *args,**kwargs:7008)
    monkeypatch.setattr(driver.base,'git',lambda *args:'' if args[0]=='status' else 'synthetic-source')
    monkeypatch.setattr(driver.base,'build_engine',lambda *args:(None,{}))
    monkeypatch.setattr(driver,'build_trace',lambda *args:(None,{}))
    monkeypatch.setattr(driver.base,'load_grid',lambda *args:None)
    monkeypatch.setattr(driver,'build_atlas_geometry',lambda *args:None)
    def fakefeatures(role,p,directory,*args):
        if role=='test':assert (directory/'model-seal.json').exists()
        events.append(role+'_features');driver.base.write(directory/f'{role}-features.json',[])
        return [],{}
    def faketargets(role,p,directory,*args):
        if role=='test':driver.authorize_test_targets(directory)
        events.append(role+'_targets');(directory/f'{role}-targets.jsonl').write_text('')
        return []
    def fakefit(*args):events.append('fit');return {'status':'complete'}
    def fakedecide(*args):events.append('decide');return []
    monkeypatch.setattr(driver,'collect_features',fakefeatures)
    monkeypatch.setattr(driver,'collect_targets',faketargets)
    monkeypatch.setattr(driver,'fit_development',fakefit)
    monkeypatch.setattr(driver,'decide_test',fakedecide)
    monkeypatch.setattr(driver,'summarize',lambda *args:{'mock':True})
    monkeypatch.setattr(driver,'verify',lambda *args:events.append('verify'))
    destination=tmp_path/'attempt'
    driver.run(destination)
    assert events==['development_features','development_targets','fit','test_features','decide','test_targets','verify']
    assert json.loads((destination/'receipt.json').read_text())['status']=='complete'
    with pytest.raises(FileExistsError):driver.run(destination)
