import sys,json,os,shutil,subprocess
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_clark_organizer as W


def test_exact_return_binding_rejects_mutated_domain():
 r=dict(dimension=26,center=[0]*26,box=[1]*26,center_value=[[2]]*26,whole_value=[[3]]*26,whole_jacobian=[[4]*26 for _ in range(26)])
 a=dict(z=r['center']);b=dict(value=r['center_value']);c=dict(z=r['box']);d=dict(value=r['whole_value'],jacobian=r['whole_jacobian']);W.root_binding(r,a,b,c,d)
 c['z']=[9]*26
 with pytest.raises(ValueError):W.root_binding(r,a,b,c,d)


def test_same_root_requires_all26_strict_memberships():
 with ctx.workprec(180):
  old=[arb(0,arb((1,-96)))]*26;fresh=[arb(0,arb((1,-94)))]*26;assert W.same_root(old,fresh)['all26_original_boxes_strictly_inside']
  with pytest.raises(ValueError):W.same_root(old,fresh[:-1])
  with pytest.raises(ValueError):W.same_root(old,old)
  fresh[-1]=arb(1,arb((1,-94)))
  with pytest.raises(ValueError):W.same_root(old,fresh)


def test_reserved_failure_and_manifest_survive_normal_cap(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=124*1024**2-10
 with pytest.raises(ValueError):w.save('too-large.json',{'message':'x'*50})
 w.save('outcome.json',dict(passed=False,error='retained cap failure'),True);w.save('manifest.json',dict(files={'outcome.json':W.entry(tmp_path/'outcome.json')}),True)
 assert json.loads((tmp_path/'outcome.json').read_text())['passed']is False


def test_inclusive_native_count_and_actual_launcher_contract():
 assert W.CAPS['native_phases']==W.CAPS['augmented']*3+W.CAPS['path5']*3+12==54
 text=(ROOT/'tools/launch_clark_organizer_review.py').read_text();assert 'tools/reviews/review_clark_organizer.py'in text and 'attempt001'in text and "'internal_seconds':60,'external_seconds':75"in text and "'worker_rss_kib':1024**2"in text and '130*1024**2'in text


def test_isolated_complete_source_archive_import(tmp_path):
 for name in W.FILES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
 code="""import sys
from pathlib import Path
import review_clark_organizer as W
root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.FILES),(loaded-set(W.FILES))
assert Path(W.__file__).resolve().is_relative_to(root)
"""
 env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools/reviews'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
