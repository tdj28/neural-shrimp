import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_clark_critical_bridge as W

def test_exact_organizer_authority_requires_every_old_and_fresh_coordinate():
 original=dict(passed=True,primitive_period=6,critical_signed_multiplier=-1,critical_native_multiplier=1)
 audit=dict(passed=True,primitive_period=6,complement_dimension=11)
 fresh=dict(passed=True,box=[{}]*26,coordinate_inclusions=[True]*26)
 same=dict(all26_original_boxes_strictly_inside=True)
 W.exact_authority(original,audit,fresh,same)
 for mutation in('one_coordinate','short_box','old_failed','membership'):
  o,a,f,s=map(copy.deepcopy,(original,audit,fresh,same))
  if mutation=='one_coordinate':f['coordinate_inclusions'][10]=False
  elif mutation=='short_box':f['box'].pop()
  elif mutation=='old_failed':o['passed']=False
  else:s['all26_original_boxes_strictly_inside']=False
  with pytest.raises(ValueError):W.exact_authority(o,a,f,s)

def test_unique_normalization_orientation_must_match_direct_and_native_return():
 def rows(sign):return [dict(sign=x,join=dict(passed=x==sign,coordinates=26))for x in(1,-1)]
 assert W.choose_orientation(rows(-1),rows(-1))==-1
 with pytest.raises(ValueError):W.choose_orientation(rows(1),rows(-1))
 bad=rows(1);bad[1]['join']['passed']=True
 with pytest.raises(ValueError):W.choose_orientation(bad,rows(1))
 bad=rows(1);bad[0]['join']['coordinates']=25
 with pytest.raises(ValueError):W.choose_orientation(bad,rows(1))

def test_full_synthetic_profile_and_inclusive_native_counts(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work()
 with ctx.workprec(384):W.synthetic(w)
 profile=json.loads((tmp_path/'profile.json').read_text());assert profile['passed']and profile['target_arrays_unopened']
 assert w.calls['family']==3 and w.calls['context']==w.calls['transport']==1 and w.calls['native_phase_returns']==21
 assert W.CAPS['family']==3+2+3+4 and W.CAPS['native_phase_returns']==3*W.CAPS['family']+6*(W.CAPS['context']+W.CAPS['transport'])
 w.calls['family']=W.CAPS['family']
 with pytest.raises(ValueError):w.charge('family')

def test_final_failure_reserve_and_actual_launcher_contract(tmp_path,monkeypatch):
 actual_output=W.OUT;monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=124*1024**2-1
 with pytest.raises(ValueError):w.save('too-large.json',dict(test='x'*20))
 w.save('outcome.json',dict(passed=False),True);assert(tmp_path/'outcome.json').is_file()
 s=(ROOT/'tools/launch_clark_critical_bridge.py').read_text();assert str(actual_output.parent)in s and 'tools/certify_clark_critical_bridge.py'in s and "'internal_seconds':90,'external_seconds':105"in s and '130*1024**2'in s and '258*1024**2'not in s
 assert json.loads((ROOT/W.CARD).read_text())['settings']==W.SETTINGS

def test_complete_archive_import_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="""import sys
from pathlib import Path
import certify_clark_critical_bridge as W
root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.FILES),loaded-set(W.FILES)
assert Path(W.__file__).resolve().is_relative_to(root)
"""
 env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)

def test_weighted_amendment_uses_only_literal_dyadic_proposals():
 from flint import arb
 d=dict(dimension=26,proposed_exponents=W.RADIUS_EXPONENTS.copy(),proposed_radii=[2.**e for e in W.RADIUS_EXPONENTS]);o=dict(passed=True,native_calls=0,status='saved_floating_diagnosis_complete',original_strict_inclusions=5)
 assert W.weighted_radii(d,o)==[arb((1,e))for e in W.RADIUS_EXPONENTS]
 for key in ['proposed_exponents','proposed_radii']:
  q=copy.deepcopy(d);q[key][5]+=1
  with pytest.raises(ValueError):W.weighted_radii(q,o)
 bad=copy.deepcopy(o);bad['native_calls']=1
 with pytest.raises(ValueError):W.weighted_radii(d,bad)
