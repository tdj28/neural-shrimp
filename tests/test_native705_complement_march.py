import copy,json,shutil,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_complement_march as P
import launch_native705_complement_march as L


def fixture():
    card=json.loads((ROOT/P.CARD).read_text());row=card['charts'][0];c=P.cell(row['settings']['parameter'],'1/128',0,0);job=L.make_assignment(1,row,c,'source');return card,job


def test_all_fixed24_same_original_units_and_canonical_cells():
    card,job=fixture();assert L.BASE==P.BASE and card['limits']==L.LIMITS
    for ordinal in range(7,31):
        row=P.M.select(card,ordinal);assert row['unit']==(ordinal-7)%8+1
        assert P.cell(row['settings']['parameter'],'1/128',0,0)['endpoints']==row['parameter_endpoints']
    assert P.assignment(card,job,1,'source')[1]==job['cell']
    for key,bad in [('ordinal',6),('source_commit','wrong'),('root','/tmp/wrong'),('attempt',2)]:
        with pytest.raises(ValueError):P.assignment(card,dict(job,**{key:bad}),1,'source')
    bad=copy.deepcopy(job);bad['cell']['center']='0'
    with pytest.raises(ValueError):P.assignment(card,bad,1,'source')


def test_actual_profile_target_counts_and_owner_are_mandatory():
    _,job=fixture();done=dict(assignment=job,unit=1,passed=True,status='whole_chart_complement_attracting',metric_passed=True,complement_dimension=11,native_calls=360,profile_native_calls=180,target_native_calls=180)
    assert L.classify(done,job,dict(exit_code=0))=='accept'
    for key,bad in [('native_calls',359),('target_native_calls',179),('profile_native_calls',179)]:assert L.classify(dict(done,**{key:bad}),job,dict(exit_code=0))=='stop'
    with pytest.raises(ValueError):L.classify(done,job,dict(exit_code=2))


def test_adapter_restores_historical_runner_bindings(monkeypatch,tmp_path):
    card,job=fixture();old={k:getattr(P.H,k)for k in ['BASE','CARD','FILES','geometry','bind','preserved_profile','save_bound']}
    monkeypatch.setattr(P,'retained_failure',lambda card:None)
    monkeypatch.setattr(P.subprocess,'check_output',lambda *a,**kw:'source\n')
    monkeypatch.setattr(P,'read',lambda path:card if path==ROOT/P.CARD else job)
    def run(unit):
        assert unit==1 and P.H.BASE==P.BASE and P.H.geometry(unit)==(job['cell']['center'],job['cell']['endpoints'])
        assert P.H.FILES==P.FILES;P.H.preserved_profile(card)
        P.H.save_bound(tmp_path,'profile.json',dict(passed=True,forecast_seconds=321,forecast_bytes=300*1024**2,native_calls=180),[0])
        assert json.loads((tmp_path/'profile.json').read_text())['complete48_forecast_bytes']<16*1024**3
        raise ValueError('retain failure')
    monkeypatch.setattr(P.H,'main',run)
    with pytest.raises(ValueError,match='retain failure'):P.main(1)
    assert all(getattr(P.H,k)==v for k,v in old.items())


def test_worker_inventory_rejects_missing_and_changed_bytes(tmp_path):
    (tmp_path/'record.json').write_text('{}');L.save(tmp_path,'manifest.json',dict(files={'record.json':L.entry(tmp_path/'record.json')}))
    assert L.verify_worker(tmp_path)['sha256']
    (tmp_path/'record.json').write_text('{ }')
    with pytest.raises(ValueError):L.verify_worker(tmp_path)


def test_isolated_source_archive_contains_executed_module_closure(tmp_path):
    for name in P.FILES:
        out=tmp_path/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
    code='import sys,pathlib;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import probe_native705_complement_march as p;assert p.ROOT==pathlib.Path('+repr(str(tmp_path))+');loaded={str(pathlib.Path(m.__file__).resolve().relative_to(p.ROOT))for m in list(sys.modules.values())if getattr(m,"__file__",None)and pathlib.Path(m.__file__).resolve().is_relative_to(p.ROOT)};assert loaded<=set(p.FILES),loaded-set(p.FILES);assert all((p.ROOT/n).is_file()for n in p.FILES)'
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)


def test_full_plan_admission_cannot_inherit_only_worker_success(monkeypatch,tmp_path):
    card,job=fixture();monkeypatch.setattr(P,'retained_failure',lambda card:None)
    monkeypatch.setattr(P.subprocess,'check_output',lambda *a,**kw:'source\n');monkeypatch.setattr(P,'read',lambda path:card if path==ROOT/P.CARD else job)
    def run(unit):
        P.H.save_bound(tmp_path,'profile.json',dict(passed=True,forecast_seconds=321,forecast_bytes=370*1024**2,native_calls=180),[0]);return 2
    monkeypatch.setattr(P.H,'main',run);assert P.main(1)==2
    assert not json.loads((tmp_path/'profile.json').read_text())['passed']


def test_all_saved_index_metadata_schemas_without_dense_arrays():
    # Literal metadata only; no matrix decode, model or interval target operation.
    card,_=fixture()
    import hashlib
    for row in card['charts']:
        run=Path(row['producer_root']);ci=json.loads((run/'chart-inputs.json').read_text());ir=Path(row['independent']['root']);io=json.loads((ir/'run/outcome.json').read_text());join=json.loads((ir/f"run/chart-{row['unit']}/join.json").read_text());uniform=json.loads((ir/f"run/chart-{row['unit']}/uniform.json").read_text())
        assert {k:hashlib.sha256(json.dumps(ci[k],sort_keys=True,separators=(',',':')).encode()).hexdigest()for k in ['normal','origin','weights']}==row['geometry_hashes']
        proof=io['proofs'][row['unit']-1];assert proof['parameter_endpoints']==row['parameter_endpoints']and proof['same_gauge']==row['same_gauge']
        assert all(v for k,v in uniform['gates'].items()if k!='anchor_identified')and uniform['inclusions']==705
        assert join['passed']and join['coordinates']==705
        assert join.get('parameter')==row['parameter_endpoints'][0]if row['unit']>1 else join['parameter_inside']


def test_literal_certificate_and_outcome_positive_angle_schemas():
    chart=dict(gates=dict(uniform_full_native_unit_pair_locus=True,primitive_period=True,realified_frame_independent=True,anchor_identified=False))
    outcome=dict(gates={'whole_angle_positive':True});uniform=dict(gates={'positive_angle':True})
    P.M.validate_chart_gates(chart,outcome,uniform)
    for which,key in [(outcome,'whole_angle_positive'),(uniform,'positive_angle')]:
        which['gates'][key]=False
        with pytest.raises(ValueError):P.M.validate_chart_gates(chart,outcome,uniform)
        which['gates'][key]=True
    assert P.BASE.name=='march002'
    card,_=fixture();assert card['retained_schema_failure']['root'].endswith('/march001')
