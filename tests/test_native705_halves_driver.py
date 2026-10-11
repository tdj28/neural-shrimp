import copy,json,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import arb
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_halves as D


def test_exact_closed_union_and_missing_or_displaced_half():
 rows=[dict(unit=i,passed=True,parameter_endpoints=list(map(str,D.Q.geometry(i)[1])))for i in(1,2)]
 assert D.closed_union(rows)==['319/4096','447/4096']
 for changed in(rows[:1],list(reversed(rows)),[rows[0],dict(rows[1],passed=False)],[rows[0],dict(rows[1],parameter_endpoints=['384/4096','447/4096'])]):
  with pytest.raises(ValueError):D.closed_union(changed)


def test_complete_owned_pair_and_failed_profile():
 row=dict(source_commit='source',passed=True,owned_live_pids=[],workers=[dict(unit=i,exit_code=0,owned_live_pids=[],worker_pid_absent=True,group_pid=90000000+i)for i in(1,2)])
 D.complete_pair(row,'source')
 for changed in(dict(row,workers=row['workers'][:1]),dict(row,passed=False),dict(row,source_commit='different')):
  with pytest.raises(ValueError):D.complete_pair(changed,'source')
 row['passed']=False
 for v in row['workers']:v['exit_code']=2
 D.complete_pair(row,'source',2)


def test_actual_path_call_mapping_without_profile_or_native_gaps(tmp_path,monkeypatch):
 def read(p):
  i=int(p.name[:3]);virtual=i-72 if i<=180 else i-180;stage=['profile_plus','profile_minus','target_plus','target_minus'][(virtual-1)//36]
  if p.name.endswith('input.json'):return dict(kind='path',stage=stage,phase=((virtual-1)%36)//2,direction='q'if i%2 else'r',inputs=[None]*15)
  return dict(kind='path',stage=stage,returns=[None]*13)
 monkeypatch.setattr(D,'read',read)
 for i in(73,108,109,144,253,288,289,324):
  kind,row=D.acquisition(tmp_path,i);assert kind=='path'and 0<=row['phase']<18


def test_exact_direction_compare_rejects_changed_native_block():
 ball=D.Q.ball_json(arb(1));row=dict(phase=0,direction='q',input=[ball],value=[[ball]],block=[[ball]],rates=[[ball]])
 D.exact_direction(row,row);bad=copy.deepcopy(row);bad['block'][0][0]=D.Q.ball_json(arb(2))
 with pytest.raises(ValueError):D.exact_direction(row,bad)


def test_fixed_card_launcher_source_and_no_native_calls():
 card=json.loads((ROOT/D.CARD).read_bytes());assert card['audit']==D.settings()and card['inputs']['units']==[1,2]and len(card['complement']['workers'])==2
 text=(ROOT/'tools/reviews/launch_native705_halves_review.py').read_text();assert str(D.OUT.parent)in text and 'review_native705_halves.py'in text and "'internal_seconds':240"in text and "'external_seconds':255"in text and '(512*1024**2,6*1024**3)'in text
 text=(ROOT/'tools/reviews/review_native705_halves.py').read_text();assert 'cusp_step('not in text and 'total_two_rate_jacobian('not in text


def test_isolated_archive_closure(tmp_path):
 for name in D.FILES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
 code="import sys;sys.path.insert(0,'tools/reviews');import review_native705_halves as d;assert all((d.ROOT/n).is_file()for n in d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
