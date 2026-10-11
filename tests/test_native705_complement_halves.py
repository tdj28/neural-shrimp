import gzip,json,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_complement_halves as H


def test_exact_closed_halves_cover_original_root_once():
    a,b=H.geometry(1),H.geometry(2)
    assert a==('351/4096',['319/4096','383/4096'])
    assert b==('415/4096',['383/4096','447/4096'])
    assert a[1][0]==H.W.geometry(1)[1][0]and a[1][1]==b[1][0]and b[1][1]==H.W.geometry(1)[1][1]
    for bad in [0,3,True,1.0]:
        with pytest.raises(ValueError):H.geometry(bad)


def test_literal_retained_wide_failure_and_original_independent_root_card():
    card=json.loads((ROOT/H.CARD).read_text())
    assert card['units']==[1,2]
    assert card['original']==json.loads((ROOT/H.W.CARD).read_text())
    assert card['failed_wide']['source']=='94faea3f137ac4a2cb5b8b63f410ed54cb42d2f2'
    import launch_native705_complement_halves as L
    assert L.BASE==H.BASE and H.BASE.name=='wide-halves002'


def test_full_mixed_input_serialization_retains_both_rates_and_gradient_dimensions():
    from neural_shrimp.rigorous_folds import MixedJet
    with ctx.workprec(384):
        x=MixedJet.variable(arb('1/32'),15,13,arb('3/64'))
        row=H.jet_json(x)
        assert len(row['g'])==len(row['m'])==15
        assert H.U.R.original_ball(row['g'][13])==1
        assert H.U.R.original_ball(row['d'])==arb('3/64')


def test_compressed_full_matrix_evidence_and_failure_reserve(tmp_path):
    used=[0];row={'complete_matrix':[[i,j]for i in range(10)for j in range(10)]}
    H.save_bound(tmp_path,'matrix.json.gz',row,used)
    assert json.loads(gzip.decompress((tmp_path/'matrix.json.gz').read_bytes()))==row
    used[0]=380*1024**2-10
    with pytest.raises(ValueError):H.save_bound(tmp_path,'overflow.json',{'v':'x'*100},used)
    H.save_bound(tmp_path,'outcome.json',{'passed':False},used,True)
    H.save_bound(tmp_path,'manifest.json',{'files':{'outcome.json':H.O.entry(tmp_path/'outcome.json')}},used,True)


def test_isolated_source_archive_contains_all_executed_local_modules(tmp_path):
    for n in H.FILES:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    code='import sys,pathlib;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import probe_native705_complement_halves as h;assert h.ROOT==pathlib.Path('+repr(str(tmp_path))+');loaded={str(pathlib.Path(m.__file__).resolve().relative_to(h.ROOT))for m in list(sys.modules.values())if getattr(m,"__file__",None)and pathlib.Path(m.__file__).resolve().is_relative_to(h.ROOT)};assert loaded<=set(h.FILES),(loaded-set(h.FILES));assert all((h.ROOT/n).is_file()for n in h.FILES)'
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)


def test_unrelated_full705_center_supports_actual_fresh_preconditioner():
    # No project target arrays. This checks the actual new profile interface,
    # all36 source-model actions, and full condensed705 preconditioner.
    with ctx.workprec(256):
        model,flat,_,_,_,normal,origin,sc,_=H.T.synthetic_inputs();calls=[]
        g,j,local=H.assemble(model,[v.mid()for v in flat],18,normal,origin,sc,tick=lambda:calls.append(1))
        C,small=H.proposal(j,local,13,flat[-1].mid())
        assert len(calls)==36 and C.nrows()==C.ncols()==705
        assert all(v.is_finite()and v.is_exact()for row in C.tolist()for v in row)


def test_resource_only_retry_retains_zero_target_failure():
    card=json.loads((ROOT/H.CARD).read_text());old=card['failed_profile']
    assert Path(old['root']).name=='wide-halves001'and old['outer']=='ac6e64d47a38c256dacb97eddcc3527721271bdec85cdaef2733e5f826aca096'
    assert len(old['workers'])==2
    assert H.BASE!=Path(old['root'])
    text=(ROOT/'tools/probe_native705_complement_halves.py').read_text()
    assert 'storage_forecast<384*1024**2'in text and 'preserved_profile(read(ROOT/CARD))'in text
