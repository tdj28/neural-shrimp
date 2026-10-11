import copy,os,sys,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_cubic34_narrow as W
import launch_cubic34_narrow_review as L

def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';c['producer']={k:'test-only'for k in c['producer']};c['producer']['root']=str(W.N.OUT.parent);return c

def test_prepared_and_missing_pins_never_admit():
 c=card();W.card_check(c);L.card_check(c);assert W.OUT.parent==L.OUT
 for key in c['producer']:
  bad=copy.deepcopy(c);bad['producer'][key]=None
  with pytest.raises(ValueError):W.card_check(bad)
  with pytest.raises(AssertionError):L.card_check(bad)
 with pytest.raises(ValueError):W.card_check(dict(c,status='prepared'))

def test_full_original_interval_and_cap():
 domains=W.N.H.partition();assert len(domains)==64 and W.N.H.CALLS==30096
 rows=[dict(cell=W.N.C.H.cell(domains,i+1),passed=True,complement_dimension=11)for i in range(64)]
 assert W.B.exact_union(rows,['83/512','115/512'])['total_cells']==64
 with pytest.raises(ValueError):W.B.exact_union(rows[:-1],['83/512','115/512'])
 with pytest.raises(ValueError):W.B.exact_union(rows+[rows[0]],['83/512','115/512'])

def mock_history():
 domains=W.N.H.partition();cursor=144;attempts=[];files={};accepted=[]
 for dom in domains[:63]:
  files[f"domains/{dom['index']:03d}/domain-outcome.json"]=dict(passed=True,domain=dom,dimension=705,native_calls=90,first_call=cursor+1,last_call=cursor+90);cursor+=90
  sequence=[(0,0),(1,0)]if dom['index']==63 else ([(0,0),(1,0),(1,1)]if dom['index']in(1,2,3,60,61,62)else[(0,0)])
  for depth,index in sequence:
   cell=W.N.C.H.cell(domains,dom['index'],depth,index);ordinal=len(attempts)+1;passed=dom['index']<63 and (len(sequence)==1 or depth==1);r=dict(cell=cell,passed=passed,completed=True,native_calls=126,dimension=705,complement_dimension=11,first_call=cursor+1,last_call=cursor+126);cursor+=126;attempts.append(r)
   if passed:accepted.append(r)
   files[f'cells/{ordinal:03d}/cell-input.json']=cell;files[f'cells/{ordinal:03d}/cell-outcome.json']=r;files[f'completed-attempt-{ordinal:03d}.json']=r;files[f'cells/{ordinal:03d}/complement.json']={'passed':passed}
 pending=[W.N.C.H.cell(domains,63,1,k)for k in(0,1)]
 out=dict(passed=False,status='failed_or_incomplete',error='ValueError: fixed maximum cell depth1',domains_done=63,pending_cells=pending,completed_attempts=attempts,accepted_cells=accepted,native_calls=cursor,profile_native_calls=144,target_native_calls=cursor-144)
 return domains,out,files

def test_history_exact_stopped_prefix_and_corruptions(monkeypatch):
 domains,out,files=mock_history();monkeypatch.setattr(W,'read',lambda p:files[str(p.relative_to(Path('/mock')))]);monkeypatch.setattr(W.A,'pairs',lambda *a:None)
 stages,cells=W.history(out,Path('/mock'),dict(domains=domains));assert len(stages)==63 and len(cells)==68 and out['native_calls']==15390
 for mutate in (lambda b:b.update(native_calls=b['native_calls']-1),lambda b:b.update(passed=True),lambda b:b.update(pending_cells=[]),lambda b:b['completed_attempts'].pop(),lambda b:b.update(accepted_cells=b['accepted_cells'][:-1]),lambda b:b.update(union=dict(passed=True))):
  bad=copy.deepcopy(out);mutate(bad)
  with pytest.raises((ValueError,KeyError)):W.history(bad,Path('/mock'),dict(domains=domains))

def test_prefix_closed_union_excludes_terminal_domain():
 _,out,_=mock_history();assert W.B.exact_union(out['accepted_cells'],W.PREFIX)['total_cells']==68
 with pytest.raises(ValueError):W.B.exact_union(out['accepted_cells'],['83/512','115/512'])
 with pytest.raises(ValueError):W.B.exact_union(out['accepted_cells'][:-1],W.PREFIX)


def test_final_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=1020*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',dict(passed=False),True)

def test_isolated_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import review_cubic34_narrow as w;import launch_cubic34_narrow_review as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES);assert w.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr


def test_launcher_blocks_prepared_before_output(monkeypatch,tmp_path):
 c=card();c['status']='prepared';monkeypatch.setattr(L.subprocess,'check_output',lambda *a,**k:b'');monkeypatch.setattr(L.json,'loads',lambda *_:c);monkeypatch.setattr(L,'OUT',tmp_path/'unopened')
 with pytest.raises(AssertionError):L.main()
 assert not L.OUT.exists()

def test_prior_whole_projection_inside_prefix():
 assert W.prefix_projection(['333/2048','333/2048'])['passed']
 for projection in (['0','333/2048'],['333/2048','115/512'],['333/2048','332/2048']):
  with pytest.raises(ValueError):W.prefix_projection(projection)
