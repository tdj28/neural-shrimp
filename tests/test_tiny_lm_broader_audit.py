"""Synthetic controls only; never opens target corpus, collection, or analysis."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_tiny_lm_broader_data as audit
import analyze_tiny_lm_broader_data as producer
from neural_shrimp.tiny_lm_prediction import CausalFeatures
from neural_shrimp.tiny_lm_progress import ridge_fit, ar_progress, progress_targets


def synthetic_records():
    rng = np.random.default_rng(761)
    names, groups = audit.feature_schema()
    records = []
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            ce = 3-.0004*np.arange(800)+.1*np.sin(np.arange(800)/13)+rng.normal(0,.03,800)
            X = rng.normal(size=(800,len(names)))
            X[:,0] = 0; X[:,1] = 1; X[:,4] = np.nan
            X[:,2] = ce; X[:,-1] = np.log(arm*32)
            records.append(dict(seed=seed,arm=arm,eta=.1,initial_state_sha256=str(seed),role='collect',
                data_identity_sha256='synthetic',outcome='completed',completed_updates=800,
                final_train_ce=float(ce[-1]-.001),X=X,ce=ce,y=audit.labels(ce),groups=groups,
                feature_groups=groups,feature_names=names,probe=[{'step':t,'status':'ok','ce':3.-t*.0003}
                    for t in [0,25,*range(100,801,100)]],elapsed_seconds=1.))
    return records


@pytest.fixture(scope='module')
def numerical_fixture():
    records = synthetic_records()
    expected = audit.analysis(records)
    observed = producer.analyze(records)
    return records, expected, observed


def test_all_candidate_fits_and_complete_report(numerical_fixture):
    _, expected, observed = numerical_fixture
    audit.compare(observed,expected)
    assert sum(len(f['candidates']) for s in expected['seeds'].values() for f in s['models'].values()) == 32


@pytest.mark.parametrize('kind',['unselected_weight','chosen_penalty','prediction','metric','gate'])
def test_semantic_analysis_mutations_rejected(numerical_fixture,kind):
    _, expected, observed = numerical_fixture
    tampered = copy.deepcopy(observed)
    seed = tampered['seeds']['28161']
    if kind == 'unselected_weight':
        selected = seed['models']['B3']['selected']['penalty']
        candidate = next(c for c in seed['models']['B3']['candidates'] if c['penalty'] != selected)
        candidate['model']['weights'][1] += .1
    elif kind == 'chosen_penalty':
        seed['models']['B3']['selected'] = next(c for c in seed['models']['B3']['candidates'] if c['penalty'] != seed['models']['B3']['selected']['penalty'])
    elif kind == 'prediction':
        seed['scores']['B2']['256']['test']['prediction'][7] += .01
    elif kind == 'metric':
        seed['scores']['AR1']['256']['test']['normalized_mse'] += .01
    else:
        tampered['positive_forecast_interpretation'] = not tampered['positive_forecast_interpretation']
    with pytest.raises(audit.AuditFailure):
        audit.compare(tampered,expected)


def test_ridge_is_independent_augmented_solver():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(81,9)); X[:,1] = np.nan; X[:,2] = 7
    X[:,3] = X[:,4]; X[:6,8] = np.nan
    y = rng.normal(size=81)
    for penalty in audit.PENALTIES:
        independent = audit.ridge(X,y,penalty)
        audit.compare(ridge_fit(X,y,penalty),independent)
    assert abs(audit.ridge(X,np.repeat(4.,81),.1)['weights'][0]-4) < 1e-12
    X[0,0] = np.inf
    with pytest.raises(ValueError): audit.ridge(X,y,.1)


def test_future_labels_and_embargo():
    ce = np.arange(800,dtype=float)
    np.testing.assert_equal(audit.labels(ce),progress_targets(ce))
    ce[395] = np.nan
    y = audit.labels(ce)
    assert np.isnan(y[379:396]).all() and np.isnan(y[-16:]).all()
    assert audit.indices(800,'fit')[-1]+16 == 399 < audit.indices(800,'tune')[0]
    assert audit.indices(800,'tune')[-1]+16 == 599 < audit.indices(800,'test')[0]
    assert audit.indices(800,'test')[-1]+16 == 799


def test_ar_rank_and_causality():
    rng = np.random.default_rng(7)
    ce = rng.normal(size=64)
    for order in (1,2):
        a = audit.ar(ce[:32],order); b = ar_progress(ce[:32],order)
        audit.compare(a,b)
        assert audit.ar(np.ones(32),order)[1] == 'rank_deficient'
        assert audit.ar(ce[:31],order)[1] == 'insufficient_history'
        ce[40:] += 100
        audit.compare(a,audit.ar(ce[:32],order))


def test_probe_and_future_test_changes_do_not_select_models(numerical_fixture):
    records, reference, _ = numerical_fixture
    altered = copy.deepcopy(records)
    for r in altered:
        r['probe'] = [{'step':800,'ce':100000.,'status':'ok'}]
        r['y'][600:784] += 20.
    result = audit.analysis(altered)
    for seed in ('28161','28162'):
        audit.compare(result['seeds'][seed]['models'],reference['seeds'][seed]['models'])
        assert result['seeds'][seed]['selected_ordinary'] == reference['seeds'][seed]['selected_ordinary']
    only_probe = copy.deepcopy(records)
    for r in only_probe: r['probe'] = []
    audit.compare(audit.analysis(only_probe)['seeds'],reference['seeds'])


def test_any_comparator_row_failure_blocks_positive(numerical_fixture):
    records, _, _ = numerical_fixture
    altered = copy.deepcopy(records)
    altered[0]['ce'][:32] = 2.
    altered[0]['y'] = audit.labels(altered[0]['ce'])
    altered[0]['X'][610,3] = np.inf
    result = audit.analysis(altered)
    assert not result['seeds']['28161']['all_comparators_and_eligible_rows_valid']
    assert not result['positive_forecast_interpretation']
    assert result['seeds']['28161']['scores']['B0']['8']['test']['failed_forecast_steps'] == [610]


def test_initialization_reconstruction_no_forward():
    from neural_shrimp.tiny_lm import TinyTransformer
    for seed in audit.SEEDS:
        np.testing.assert_array_equal(audit.initial_state(seed),TinyTransformer(66).initialize(seed).numpy())


def test_corpus_roles_against_direct_bytes_and_tamper():
    raw = bytes((65+i%3 for i in range(205000)))
    roles = audit.expected_roles(raw,['A','B','C','<UNK>'])
    assert roles['expanded']['window_token_ids'][:8] == roles['control']['window_token_ids']
    assert roles['probe']['starts'][0] == 196608
    altered = bytearray(raw); altered[65536] = 67 if raw[65536] != 67 else 65
    changed = audit.expected_roles(bytes(altered),['A','B','C','<UNK>'])
    assert changed['expanded']['selected_bytes_sha256'] != roles['expanded']['selected_bytes_sha256']
    assert changed['control']['window_token_ids'] != roles['control']['window_token_ids']
    assert changed['probe'] == roles['probe']


def test_manifest_hash_inventory_and_symlink_controls(tmp_path):
    (tmp_path/'data').write_bytes(b'abc')
    m = {'files':[{'path':'data','bytes':3,'sha256':audit.sha(tmp_path/'data')}]}
    assert audit.inventory(tmp_path,m)['files'] == 1
    (tmp_path/'extra').write_text('unexpected')
    with pytest.raises(audit.AuditFailure): audit.inventory(tmp_path,m)
    (tmp_path/'extra').unlink(); (tmp_path/'data').write_bytes(b'abd')
    with pytest.raises(audit.AuditFailure): audit.inventory(tmp_path,m)
    with pytest.raises(audit.AuditFailure): audit.inventory(tmp_path,{'files':m['files']*2})
    (tmp_path/'data').unlink(); (tmp_path/'data').symlink_to('/dev/null')
    with pytest.raises(audit.AuditFailure): audit.inventory(tmp_path,m)


def test_source_inventory_cannot_drop_required_file(monkeypatch):
    monkeypatch.setattr(audit,'git_bytes',lambda commit,path:b'same')
    good = {'commit':'declared','sha256':{'a':audit.digest(b'same'),'b':audit.digest(b'same')}}
    audit.verify_source(good,'declared',['a','b'])
    with pytest.raises(audit.AuditFailure): audit.verify_source({'commit':'declared','sha256':{'a':audit.digest(b'same')}},'declared',['a','b'])
    with pytest.raises(audit.AuditFailure): audit.verify_source(good,'different',['a','b'])


def test_saved_feature_algebra_and_33_state_sufficiency():
    rng = np.random.default_rng(99)
    states = rng.normal(size=(802,7)).astype('float32')
    gradients = rng.normal(size=(801,7)).astype('float32')
    ce = 3+rng.normal(size=801)*.1
    monitor = CausalFeatures(); rows = []
    for t in range(801):
        rows.append(np.r_[monitor.observe(ce[t],gradients[t],states[t],None if t == 0 else states[t]-states[t-1]),np.log(256)])
    r = {'X':np.array(rows),'ce':ce,'arm':8,'feature_names':monitor.names+['log_training_targets']}
    assert audit.saved_feature_checks(r,states)['rows_from_step32'] == 769
    for end in audit.SAMPLE_STEPS:
        short = CausalFeatures()
        for t in range(end-32,end+1):
            value = short.observe(ce[t],gradients[t],states[t],None if t == 0 else states[t]-states[t-1])
        np.testing.assert_allclose(value,r['X'][end,:-1],rtol=0,atol=0,equal_nan=True)
    r['X'][601,r['feature_names'].index('plane_coherence')] += .1
    with pytest.raises(AssertionError): audit.saved_feature_checks(r,states)


def test_exclusive_attempt_and_raw_cap_preserve_failure(tmp_path,monkeypatch):
    audit.write_exclusive(tmp_path/'stage.json',{'preserved':True})
    with pytest.raises(FileExistsError): audit.write_exclusive(tmp_path/'stage.json',{})
    monkeypatch.setattr(audit,'CAP',40)
    with pytest.raises(audit.AuditFailure): audit.write_exclusive(tmp_path/'too-large.json',{'x':'y'*40})
    assert json.loads((tmp_path/'stage.json').read_text()) == {'preserved':True}


def test_deadline_retains_stage_and_failure_manifest(tmp_path,monkeypatch):
    root = tmp_path/'source'; root.mkdir()
    paths = ['dummy.py','src/neural_shrimp/tiny_lm.py','src/neural_shrimp/tiny_lm_prediction.py']
    for name in paths:
        p = root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture')
    monkeypatch.setattr(audit,'ROOT',root)
    monkeypatch.setattr(audit,'AUDIT_FILES',paths)
    monkeypatch.setattr(audit,'git_bytes',lambda commit,path:(root/path).read_bytes())
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**k:'fake-commit\n')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        monkeypatch.setenv(name,'1')
    output = tmp_path/'audit'
    monkeypatch.setattr(sys,'argv',['audit','--collection',str(tmp_path/'input'),'--analysis',str(tmp_path/'analysis.json'),
        '--corpus',str(tmp_path/'corpus'),'--output',str(output),'--producer-commit','fake',
        '--manifest-sha256','fake','--analysis-sha256','fake'])
    def timeout(args,out,checkpoint):
        checkpoint('partial',{'retained':True})
        raise audit.AuditDeadline('synthetic deadline')
    monkeypatch.setattr(audit,'audit',timeout)
    assert audit.main() == 1
    result = json.loads((output/'result.json').read_text())
    assert not result['passed'] and result['completed_stages'] == ['request','partial']
    assert (output/'stage-01-partial.json').exists()
    assert audit.inventory(output,json.loads((output/'manifest.json').read_text()))['files'] == 3


def test_invalid_budget_rejects_before_attempt_creation(tmp_path,monkeypatch):
    out = tmp_path/'audit'
    monkeypatch.setattr(sys,'argv',['audit','--collection',str(tmp_path/'input'),'--analysis','unused',
        '--corpus',str(tmp_path/'corpus'),'--output',str(out),'--producer-commit','fake',
        '--manifest-sha256','fake','--analysis-sha256','fake','--max-seconds','nan'])
    with pytest.raises(audit.AuditFailure): audit.main()
    assert not out.exists()


def test_fully_rehashed_token_substitution_fails_original_corpus_binding(tmp_path,monkeypatch):
    corpus = tmp_path/'corpus'; corpus.mkdir()
    collection = tmp_path/'collection'; collection.mkdir()
    raw = bytes(32+i%65 for i in range(1115394))
    (corpus/'input.txt').write_bytes(raw)
    original_sha = audit.digest(raw)
    monkeypatch.setattr(audit,'CORPUS',original_sha)
    monkeypatch.setattr(producer,'CORPUS_SHA256',original_sha)
    original = {'corpus_sha256':original_sha,'bytes':len(raw)}
    (corpus/'receipt.json').write_text(json.dumps(original))
    vocabulary = sorted(set(raw[:1003854].decode('ascii')))+['<UNK>']
    d = {'source':original,'corpus_sha256':original_sha,'corpus_bytes':len(raw),
         'vocabulary':vocabulary,'vocabulary_sha256':audit.canonical(vocabulary),
         'roles':audit.expected_roles(raw,vocabulary),
         'target_rule':'X=chars[s:s+32]; Y=chars[s+1:s+33], next character at every position'}
    def save():
        d['data_identity_sha256'] = audit.canonical({k:d[k] for k in
            ('corpus_sha256','corpus_bytes','vocabulary','roles','target_rule')})
        (collection/'data.json').write_text(json.dumps(d))
        arrays = {}
        for role, r in d['roles'].items():
            tokens = np.array(r['window_token_ids'],dtype='<i8')
            arrays[role+'_X'], arrays[role+'_Y'] = tokens[:,:-1], tokens[:,1:]
        np.savez(collection/'data-tokens.npz',**arrays)
    save()
    audit.verify_corpus(collection,corpus)
    for role in ('expanded','control'):
        metadata = d['roles'][role]
        metadata['window_token_ids'][0][0] = (metadata['window_token_ids'][0][0]+1)%65
        metadata['window_tokens_sha256'] = audit.digest(np.array(metadata['window_token_ids'],dtype='<i8').tobytes())
    save()
    producer.verify_data(collection)  # Producer internal-consistency check accepts this rehashed substitution.
    with pytest.raises(audit.AuditFailure,match='original corpus window/token mismatch'):
        audit.verify_corpus(collection,corpus)
