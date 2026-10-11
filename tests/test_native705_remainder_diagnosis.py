import sys
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_native705_remainder as D
import launch_native705_remainder_diagnosis as L

def test_phase_uncertainty_decomposition_does_not_drop_center_floor():
 with ctx.workprec(192):
  q=arb_mat([[arb(1,arb(1)/32)]]);p=arb_mat([[arb(2,arb(1)/8)]])
  c,e0,e1=D.reduce_phase(q,p,arb(1)/4)
  assert e0[0,0].contains(arb(1)/32)and e1[0,0].contains(arb(1)/32)
  cc,x,y=D.reduce_phase(q,p,arb(0));assert D.ball_json(x[0,0])==D.ball_json(e0[0,0]) and y[0,0].is_zero()and cc[1][0,0].is_zero()

def test_exact_scope_and_resources():
 assert D.OUT.parent==L.OUT
 r=D.read(ROOT/'docs/reviews/native705-polynomial-complement-attempt001.json');assert r['outer']==D.PIN and r['source']==D.EXEC and r['outcome']['native_calls']==216
 s=Path(L.__file__).read_text();assert "'internal_seconds':30"in s and "'external_seconds':45"in s and "'output_cap_bytes':10*1024**2"in s


def test_archive_executes_without_unretained_repo_modules(tmp_path):
 import shutil,subprocess
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 script="import sys,pathlib;sys.path.insert(0,'tools');import diagnose_native705_remainder as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',script],cwd=tmp_path,check=True,capture_output=True)
