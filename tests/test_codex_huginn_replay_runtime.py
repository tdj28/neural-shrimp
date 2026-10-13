"""Synthetic runtime controls; real target/producer/model code is never used."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('task8_replay_tests',ROOT/'codex/huginn-independent-replay/replay.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def card():
    directory=Path(r.__file__).parent
    return {'schema':'codex-huginn-independent-replay-v1','status':'accepted_by_root',
            'acceptance_reference':'synthetic test only','part':'B','attempt':'part-B-001',
            'envelope':dict(r.ENVELOPE),'source_commit':'a'*40,'max_input_bytes':10000000,
            'sources':{name:r.sha(directory/name) for name in r.SOURCES},
            'inputs':{'path':'/synthetic/inputs.json','bytes':1,'sha256':'0'*64},
            'local_receipt_review_reference':'synthetic explicit authority'}


def producer_args(part):
    return ({'command':'items','start':0,'count':64,'split':'test','seed':0,
             'lams':[1.0,1.05,1.08,1.1],'fixed_tol':1e-4,'boot':2000} if part=='A' else
            {'command':'generation','start':0,'count':200,'split':'test','steps':32,'prompt_style':'cot8',
             'seed':0,'arms':['1.0','1.05','1.15','1.20'],'base':'1.0','collapse_arm':'1.20',
             'break_arm':'1.15','safe_arm':'1.05','margin':.04,'max_new':256})


def test_valid_card_requires_all_four_sources():
    r.validate_card(card(),Path(r.__file__).parent)


@pytest.mark.parametrize('field,value',[('status','prepared'),('part','prefix'),('attempt','part-B-002'),
    ('acceptance_reference',''),('source_commit','not-a-hash'),('max_input_bytes',True),
    ('local_receipt_review_reference','')])
def test_unreviewed_or_unbounded_cards_refused(field,value):
    c=card();c[field]=value
    with pytest.raises(ValueError):r.validate_card(c,Path(r.__file__).parent)


@pytest.mark.parametrize('value',[True,1.0,2])
def test_strict_integer_envelope(value):
    c=card();c['envelope']['cpu_threads']=value
    with pytest.raises(ValueError):r.validate_card(c,Path(r.__file__).parent)


@pytest.mark.parametrize('part',['A','B'])
def test_exact_frozen_producer_invocation(part):
    args=producer_args(part);args['out']='synthetic-output'
    assert r.validate_producer_args({'args':args},part)['count']==(64 if part=='A' else 200)
    for key in ('count','seed','split'):
        bad=copy.deepcopy(args);bad[key]=None
        with pytest.raises(ValueError):r.validate_producer_args({'args':bad},part)


def test_bool_invocation_seed_and_prefix_count_refused():
    for key,value in [('seed',False),('count',100),('steps',31),('margin',.05)]:
        args=producer_args('B');args[key]=value
        with pytest.raises(ValueError):r.validate_producer_args({'args':args},'B')


@pytest.mark.parametrize('failure',[False,True])
def test_orchestration_retains_whole_output_or_failure(tmp_path,monkeypatch,failure):
    # The synthetic admission seam deliberately avoids any real filesystem
    # authority; admission.py has separate real tar/receipt controls.
    fixture_spec=importlib.util.spec_from_file_location('task8_synthetic_profile',Path(r.__file__).parent/'profile.py')
    fixture_module=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture_module)
    reducer=r._load(Path(r.__file__).parent/'statistics.py','task8_synthetic_reducer')
    _,_,rows=fixture_module.fixtures();producer=reducer.analyze_b(rows);producer.pop('all_rows_retained');producer['args']=producer_args('B')
    fake_manifest=tmp_path/'inputs.json';fake_manifest.write_text('{}')
    class SyntheticAdmission:
        DATA_ROOT=tmp_path
        @staticmethod
        def pinned_path(pin,root):return fake_manifest
        @staticmethod
        def authenticate(*args,**kwargs):
            if failure:raise ValueError('retained synthetic admission refusal')
            return {'rows':rows,'spectra':None,'producer':producer,'manifest':{},'report':{'file_count':3}}
        @staticmethod
        def recheck(*args,**kwargs):pass
    original=r._load
    monkeypatch.setattr(r,'_load',lambda p,n:SyntheticAdmission if p.name=='admission.py' else original(p,n))
    monkeypatch.setattr(r,'OWNED',tmp_path/'owned');monkeypatch.setattr(r,'available',lambda:32*r.GIB)
    monkeypatch.setattr(r.os,'getpriority',lambda *args:10)
    monkeypatch.setattr(r.resource,'getrlimit',lambda *_:(4*r.GIB,4*r.GIB))
    c=card();p=tmp_path/'card.json';p.write_text(json.dumps(c))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):monkeypatch.setenv(key,'1')
    monkeypatch.setenv('TASK8_CARD_SHA256',r.sha(p));monkeypatch.setenv('TASK8_SOURCE_COMMIT',c['source_commit'])
    monkeypatch.setattr(sys,'argv',['replay.py','--card',str(p)])
    if failure:
        with pytest.raises(ValueError,match='synthetic admission refusal'):r.main()
    else:r.main()
    out=tmp_path/'owned'/'part-B-001';status=json.loads((out/'status.json').read_text())
    assert status['computational_completed'] is (not failure)
    assert status['producer_or_model_executed'] is False and status['certificate_passed'] is False
    assert (out/'card.json').exists() and all((out/name).exists() for name in r.SOURCES)
    if not failure:
        assert json.loads((out/'independent.json').read_text())['n']==200
        assert (out/'comparison.json').exists() and (out/'producer.json').exists()
    with pytest.raises(FileExistsError):r.main()
