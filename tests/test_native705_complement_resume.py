import copy,json,shutil,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_complement_resume as P
import launch_native705_complement_resume as L
import resume_native705_complement_binding as R

def fixture():
    card=json.loads((ROOT/P.CARD).read_text());old=R.prior(card,full=False);ordinal,c=old['pending'][0];row=P.M.select(card,ordinal);return card,L.make_assignment(1,row,c,'source'),old

def test_literal_pending_excludes_previously_passed_cells_and_changes():
    card,job,old=fixture();assert P.assignment(card,job,1,'source')[1]==job['cell'];assert len(old['pending'])==8 and len(R.accepted(old)[7])==1
    for ordinal,depth,index in [(7,0,0),(23,1,0),(26,0,0),(31,0,0)]:
        bad=copy.deepcopy(job);bad['ordinal']=ordinal
        if ordinal<=30:bad['cell']=P.cell(P.M.select(card,ordinal)['settings']['parameter'],'1/128',depth,index)
        with pytest.raises(ValueError):P.assignment(card,bad,1,'source')
    for depth,index in [(2,2),(2,3),(3,4),(3,7)]:
        descendant=copy.deepcopy(job);descendant['cell']=P.cell(P.M.select(card,26)['settings']['parameter'],'1/128',depth,index);assert P.assignment(card,descendant,1,'source')
    bad=copy.deepcopy(card);bad['resume']['outer']['sha256']='0'*64
    with pytest.raises(ValueError):R.prior(bad,full=False)
    bad=copy.deepcopy(card);bad['charts'][0]['geometry_hashes']['normal']='0'*64
    with pytest.raises(ValueError):R.prior(bad,full=False)

def test_limits_roots_and_source_archive_closure(tmp_path):
    card,_,_=fixture();assert L.BASE==P.BASE and card['limits']==L.LIMITS and L.LIMITS['attempts']==24 and L.LIMITS['total_raw_bytes']==8*1024**3
    for name in P.FILES:
        out=tmp_path/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
    code='import sys,pathlib;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import probe_native705_complement_resume as p;loaded={str(pathlib.Path(m.__file__).resolve().relative_to(p.ROOT))for m in list(sys.modules.values())if getattr(m,"__file__",None)and pathlib.Path(m.__file__).resolve().is_relative_to(p.ROOT)};assert loaded<=set(p.FILES),loaded-set(p.FILES);assert all((p.ROOT/n).is_file()for n in p.FILES)'
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_fresh24_profile_admission_and_restoration(monkeypatch,tmp_path):
    card,job,_=fixture();old={k:getattr(P.H,k)for k in ['BASE','CARD','FILES','geometry','bind','preserved_profile','save_bound']}
    monkeypatch.setattr(P.subprocess,'check_output',lambda *a,**kw:'source\n');monkeypatch.setattr(P,'read',lambda path:card if path==ROOT/P.CARD else job)
    def run(unit):
        P.H.save_bound(tmp_path,'profile.json',dict(passed=True,forecast_seconds=399,forecast_bytes=370*1024**2,native_calls=180),[0]);return 2
    monkeypatch.setattr(P.H,'main',run);assert P.main(1)==2
    got=json.loads((tmp_path/'profile.json').read_text());assert not got['passed']and got['complete24_forecast_bytes']>8*1024**3
    assert all(getattr(P.H,k)==v for k,v in old.items())

def test_same_original_coverage_accepts_only_exact_remaining_halves():
    card,job,old=fixture();accepted=R.accepted(old);row=P.M.select(card,26)
    assert not R.coverage(row['settings']['parameter'],'1/128',accepted[26])['complete']
    accepted[26].append(job['cell']);assert R.coverage(row['settings']['parameter'],'1/128',accepted[26])['complete']
    with pytest.raises(ValueError):R.coverage(row['settings']['parameter'],'1/128',accepted[26]+[job['cell']])
