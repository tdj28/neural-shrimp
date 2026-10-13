"""Synthetic-only Task11 controls; never open actual scores or call a model."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

SOURCE=Path(os.environ.get('TASK11_SOURCE_DIR',str(Path(__file__).resolve().parents[1]/'codex/p1-second-classifier-audit')))
sys.path.insert(0,str(SOURCE))
import definitions as D
import analysis as A
import inputs as I
import task6_reducers as R
import task6_detection as F


def test_literal_complete44_and_filename_contract():
    assert len(D.ALL)==len(set(D.ALL))==44
    assert len(D.CAPTURE)==12 and len(D.HISTORY)==14 and len(D.L6)==len(D.L10)==7
    assert 'turns.npz' in D.filenames(D.L10[0])
    assert 'in-context.npz' in D.filenames(D.TEST[0])
    assert 'turns.npz' not in D.filenames(D.TEST[0]) # unused NaN probes never loaded
    with pytest.raises(ValueError):D.filenames('p1-committor')


def test_agreement_strict_threshold_and_denominators():
    a=np.array([[.5,.6,.9,.1]]);b=np.array([[.5,.2,.8,.9]])
    result=A.agreement(a,b)
    assert set(result['counts'].values())=={1}
    assert result['agreement']==.5 and result['first_only']==result['second_only']==.25
    assert result['turns']==4 and result['share_difference_second_minus_first']==0


def test_classifier_identity_shape_and_nonfinite_refuse():
    labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    a=np.zeros((2,40,6),np.float32);b=np.zeros((2,40),np.float32)
    I.validate_scores(a,labels,b,D.SECOND,D.SECOND_REVISION,2)
    for change in ['revision','nonfinite','shape','precision']:
        bb=b.copy();revision=D.SECOND_REVISION
        if change=='revision':revision='wrong'
        if change=='nonfinite':bb[0,0]=np.nan
        if change=='shape':bb=bb[:1]
        if change=='precision':bb=bb.astype(float)
        with pytest.raises(ValueError):I.validate_scores(a,labels,bb,D.SECOND,revision,2)


def test_complete_authority_refuses_missing_duplicate_changed_domain(monkeypatch):
    # Gate contract is exercised without touching actual raw data.
    manifest={'status':'complete_immutable','writer_closed_reference':'synthetic',
              'arms':list(D.ALL),'files':[],'sample_args':{a:{} for a in D.ALL}}
    with pytest.raises(ValueError,match='identity'):I.authority(manifest)
    manifest['arms']=list(D.ALL[:-1])+[D.ALL[0]]
    with pytest.raises(ValueError,match='arm order'):I.authority(manifest)
    manifest['status']='writing'
    with pytest.raises(ValueError,match='writer-complete'):I.authority(manifest)


def test_complete_synthetic44_manifest_success_then_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(I,'RAW',tmp_path)
    monkeypatch.setattr(I,'validate_producer',lambda m:None) # isolated file-identity layer; actual closure below
    records=[]
    for a in D.ALL:
        (tmp_path/a).mkdir()
        for name in D.filenames(a):
            p=tmp_path/a/name;p.write_bytes(b'synthetic metadata only')
            records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':I.digest(p)})
    manifest={'status':'complete_immutable','writer_closed_reference':'synthetic test closure',
              'arms':list(D.ALL),'files':records,'sample_args':{a:{} for a in D.ALL}}
    assert I.authority(manifest) is manifest
    Path(records[-1]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):I.authority(manifest)


def test_literal_closed_producer44_metadata_binding():
    # The committed closure manifest contains hashes only, no probability arrays.
    p=json.loads((SOURCE/'producer-score-manifest.json').read_text())
    m={'producer_manifest_sha256':I.PRODUCER_MANIFEST_SHA,'producer_source_commit':I.PRODUCER_SOURCE,
       'files':[{**r,'path':str(Path('/data2/neural-shrimp-data')/r['path'])} for r in p['files']]}
    I.validate_producer(m)
    m['files'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='closed producer44'):I.validate_producer(m)


def test_capture_onset_conditioning_and_transition_origins():
    score=np.zeros((2,40));score[0,2:4]=1
    out=R.capture_row(score)
    assert out['never']==1 and out['median_onset_conditional_zero_based']==2
    assert out['median_onset_censored_at_end']==21
    assert out['stay']=={'numerator':1,'denominator':2,'probability':.5}


def test_event_censor_no_sentinel_and_no_future_feature():
    score=np.zeros((3,8));score[0,7]=1;score[1,4]=1
    arm={'score':score,'axis':np.arange(24).reshape(3,8),'pre':np.arange(3*8*24).reshape(3,8,24),
         'dose':.04,'feature_set':'baseline_state'}
    tab=F.make_table([arm],3,1,'baseline_state')
    assert not tab['y'][tab['group']==2].any()
    assert tab['turn'].max()==4 and tab['extra'].shape[1]==0
    for idx,(g,t) in enumerate(zip(tab['group'],tab['turn'])):
        assert tab['base'][idx,0]==score[g,t]
        np.testing.assert_array_equal(tab['state'][idx],arm['pre'][g,t+1])


def test_l10_shown_never_imputes_regenerate_originals():
    data={a:{'score':np.zeros((2,40)),'counter':np.zeros((2,40)),
             'regenerated':np.ones((2,40),bool)} for a in D.L10}
    result=A.l10_shown(data,draws=10)
    for row in result.values():
        assert row['shown_hostile_share']==0 and row['regenerated_5_14']==1
        assert row['would_be_second_classifier'].startswith('unavailable')
        assert not any(k.startswith('would_be_') and isinstance(v,(int,float)) for k,v in row.items())


def test_dialogue_weighted_auc_equals_literal_resampling():
    y=np.array([0,1,0,1,0,1]);s=np.array([.3,.3,.4,.7,.2,.9]);groups=np.array([0,0,1,1,2,2])
    chosen=np.array([2,2,0]);w=np.bincount(chosen,minlength=3)[groups]
    ids=np.concatenate([np.where(groups==g)[0] for g in chosen])
    assert R.RankedAUC(y,s).evaluate(w)==R.RankedAUC(y[ids],s[ids]).evaluate()


def test_frozen_training_transform_does_not_fit_test(monkeypatch):
    rng=np.random.default_rng(33)
    tab={'state':rng.normal(size=(60,24)),'base':rng.normal(size=(60,6)),
         'y':np.arange(60)%2}
    diag=[];fitted=F.frozen_fit(tab,diag)
    before=fitted[0].components_.copy();coefs=[m.coef_.copy() for _,m in fitted[1]]
    test={'state':rng.normal(size=(20,24))+100,'base':rng.normal(size=(20,6))+100}
    F.frozen_score(fitted,test)
    np.testing.assert_array_equal(before,fitted[0].components_)
    for coef,(_,model) in zip(coefs,fitted[1]):np.testing.assert_array_equal(coef,model.coef_)
    assert len(diag)==3 and not any(d.get('warnings') for d in diag)


def test_no_events_keeps_unestimable_primary_without_fitting(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('must not fit one-class target')
    monkeypatch.setattr(A,'frozen_fit',forbidden)
    data={a:{'first':np.zeros((2,8)),'second':np.zeros((2,8)),
             'axis':np.zeros((2,8)),'pre':np.zeros((2,8,24)),'dose':.04,
             'feature_set':'baseline_state'} for a in D.TRAIN+D.TEST}
    result,arrays=A.detection(data,'second',10)
    assert result['admissible'] is False and result['positive_pooled_lower_bound'] is None
    assert set(arrays)=={'y','group','arm','turn'}


def test_empty_test_keeps_unestimable_without_sklearn_scoring(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('must not fit or score empty pooled test')
    monkeypatch.setattr(A,'frozen_fit',forbidden);monkeypatch.setattr(A,'frozen_score',forbidden)
    data={a:{'first':np.zeros((2,8)),'second':np.zeros((2,8)),
             'axis':np.zeros((2,8)),'pre':np.zeros((2,8,24)),'dose':.04,
             'feature_set':'baseline_state'} for a in D.TRAIN+D.TEST}
    for a in D.TRAIN:data[a]['second'][0,3]=1
    for a in D.TEST:data[a]['second'][:,0]=1
    result,arrays=A.detection(data,'second',10)
    assert result['reason']=='zero observed at-risk test rows' and result['admissible'] is False
    assert result['positive_pooled_lower_bound'] is None and len(arrays['y'])==0


def test_unconverged_fit_retains_scores_but_no_scientific_verdict(monkeypatch):
    data={a:{'first':np.zeros((2,8)),'second':np.zeros((2,8)),
             'axis':np.zeros((2,8)),'pre':np.zeros((2,8,24)),'dose':.04,
             'feature_set':'baseline_state'} for a in D.TRAIN+D.TEST}
    for a in data:data[a]['second'][0,3]=1
    def fit(t,diag):diag.append({'iterations':[5000],'warnings':['synthetic convergence failure']});return None
    monkeypatch.setattr(A,'frozen_fit',fit)
    monkeypatch.setattr(A,'frozen_score',lambda f,t:([t['y'].astype(float)]*2,[.5,.5]))
    result,arrays=A.detection(data,'second',10)
    assert result['warnings']==['synthetic convergence failure'] and 'baseline' in arrays
    assert not result['admissible'] and not result['all_logistic_converged']
    assert result['positive_pooled_lower_bound'] is None


def test_reused_source_integrity():
    import hashlib
    provenance=json.loads((SOURCE/'reuse-provenance.json').read_text())
    assert hashlib.sha256((SOURCE/'task6_reducers.py').read_bytes()).hexdigest()==provenance['task6_reducers.py']['original_sha256']
    original=(SOURCE/'task6_detection.py').read_text().replace('from task6_reducers import ','from reducers import ')
    assert hashlib.sha256(original.encode()).hexdigest()==provenance['task6_detection.py']['original_sha256']


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,SOURCE/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_resource_floors_and_prepared_card_refusal(tmp_path):
    L=module('task11_launch','launch.py')
    assert L.START_AVAILABLE==20*1024**3 and L.RUN_AVAILABLE==16*1024**3
    with pytest.raises(MemoryError):L.available_gate(19*1024**3)
    with pytest.raises(MemoryError):L.available_gate(15*1024**3,True)
    L.available_gate(20*1024**3);L.available_gate(16*1024**3,True)
    W=module('task11_run','run.py')
    with pytest.raises(ValueError,match='checkpoint'):W.configure({'status':'prepared'})


def test_real_tiny_owned_child_termination():
    import subprocess
    L=module('task11_lifecycle','launch.py')
    child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        identity=L.process(child.pid);L.stop_owned(child,identity)
        assert child.poll() is not None
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)


def test_isolated_cli_help_without_models(tmp_path):
    import subprocess
    for file in ('run.py','launch.py'):
        p=subprocess.run([sys.executable,'-I','-B',str(SOURCE/file),'--help'],capture_output=True,text=True,timeout=5)
        assert p.returncode==0,p.stderr


def test_complete_isolated_scientific_import_closure(tmp_path):
    import shutil,subprocess
    W=module('task11_source_closure','run.py')
    for name in W.SOURCES:shutil.copy2(SOURCE/name,tmp_path/name)
    script='import sys;sys.path.insert(0,sys.argv[1]);import analysis,inputs;assert "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-I','-B','-c',script,str(tmp_path)],capture_output=True,text=True,timeout=5)
    assert p.returncode==0,p.stderr


def test_missing_profile_refuses_before_target_arrays():
    W=module('task11_profile_refusal','run.py')
    with pytest.raises(ValueError,match='profile authority'):W.verify_profile({'mode':'descriptive'})
