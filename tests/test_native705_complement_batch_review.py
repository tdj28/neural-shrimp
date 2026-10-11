import copy,json,shutil,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_complement_batch as D


def test_exact_four_successor_geometry_and_first_unchanged():
 assert D.H.geometry(0)==('31/4096',['-1/4096','63/4096'])
 for i in range(1,5):
  c,ends=D.geometry(i);assert c==f'{31+64*i}/4096'and ends==[f'{-1+64*i}/4096',f'{63+64*i}/4096']
 for i in(0,5,True,'1'):
  with pytest.raises(ValueError):D.geometry(i)


def test_actual216_acquisition_order_around_profile_gap():
 for i in (1,36,37,72,109,144,145,180):
  virtual=i if i<=72 else i-36;stage=['profile_plus','profile_minus','target_plus','target_minus'][(virtual-1)//36]
  a=dict(kind='path',stage=stage,phase=((virtual-1)%36)//2,direction='q'if i%2 else'r',inputs=[None]*15);b=dict(kind='path',stage=stage,returns=[None]*13)
  r=D.path_event(a,b,i);assert r['phase']==a['phase']
  for change in(dict(a,kind='synthetic'),dict(a,phase=99),dict(a,stage='wrong')):
   with pytest.raises(ValueError):D.path_event(change,b,i)
 for i in (0,73,108,181,216):
  with pytest.raises(ValueError):D.path_event({}, {},i)


def test_exact_ordered_batch_and_negative_missing_workers():
 row=dict(source_commit='source',passed=True,owned_live_pids=[],workers=[dict(unit=i,exit_code=0,owned_live_pids=[],group_pid=90000000+i)for i in range(1,5)])
 D.batch_done(row,'source')
 for key,val in [('passed',False),('owned_live_pids',[1]),('source_commit','other'),('workers',row['workers'][:3]),('workers',list(reversed(row['workers'])))]:
  with pytest.raises(ValueError):D.batch_done(dict(row,**{key:val}),'source')
 old=dict(row,commit='source',connected_count=4,connected_right='319/4096',workers=[dict(v,group=v['group_pid'])for v in row['workers']]);D.batch_done(old,'source',True)
 with pytest.raises(ValueError):D.batch_done(dict(old,connected_count=3),'source',True)


def test_fixed_input_card_launcher_and_no_new_native_calls():
 c=json.loads((ROOT/D.CARD).read_text());assert c['audit']==D.fixed_settings()and c['original']['units']==[1,2,3,4]
 assert len(c['complement']['workers'])==len(c['complement']['proofs'])==4
 assert c['complement']['outer']=='68573b8c3b02939b948485629f85aa50dc08b8ccef5f6a2bdb337868276a5dd5'
 s=(ROOT/'tools/reviews/launch_native705_complement_batch_review.py').read_text();assert "'internal_seconds':240"in s and "'external_seconds':255"in s and str(D.OUT.parent)in s and 'review_native705_complement_batch.py'in s
 s=(ROOT/'tools/reviews/review_native705_complement_batch.py').read_text();assert 'cusp_step('not in s and 'total_two_rate_jacobian('not in s


def test_isolated_complete_source_archive(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code='import sys;sys.path.insert(0,'+repr(str(tmp_path/'tools/reviews'))+');import review_native705_complement_batch as d;assert d.ROOT==__import__("pathlib").Path('+repr(str(tmp_path))+');assert all((d.ROOT/n).is_file() for n in d.FILES)'
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
