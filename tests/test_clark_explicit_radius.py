import sys,subprocess,tempfile
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_clark_explicit_radius as D

def fixture():
 n=51;h=arb(1)/1024;c=[arb(0)]*n;J=D.eye(n);zero=arb_mat(n,1);b=D.col([arb(1)]*n);p=[arb(0,h)]*n
 center=dict(input=D.vector_json(c),squared_amplitude=D.ball_json(h),jacobian=D.matrix_json(J),value=D.matrix_json(zero),parameter_derivative=D.matrix_json(b));predictor=dict(input=D.vector_json(p),squared_amplitude=D.ball_json(arb(h,h)),jacobian=D.matrix_json(J),value=D.matrix_json(zero),parameter_derivative=D.matrix_json(b));return center,predictor,[h]*n,dict(halfwidth=D.ball_json(h),left=D.ball_json(arb(0)))

def test_residual_and_anchor_failures_are_distinct_and_complete():
 with ctx.workprec(384):
  c,p,k,d=fixture();out=D.diagnose(c,p,k,d);assert not out['reproduced_radius_rejection']
  p['parameter_derivative'][0][0]=D.ball_json(arb(10));out=D.diagnose(c,p,k,d);assert out['violating_coordinates']==[0]and D.decode(out['largest'][0]['anchor_requirement'])==0
  c,p,k,d=fixture();k[26]+=2;out=D.diagnose(c,p,k,d);assert out['violating_coordinates']==[26]and D.decode(out['largest'][0]['residual_requirement'])==0
  with pytest.raises(ValueError):D.diagnose(c,p,k[:-1],d)

def test_exact_original_metadata_and_all_five_failed_proposals():
 for p,pin in zip([D.PARENT,D.PARENT/'run'],D.PINS):assert D.entry(p/'manifest.json')['sha256']==pin
 out=D.read(D.PARENT/'run/outcome.json');c=D.read(D.PARENT/'run/continuation.json');assert not out['passed']and out['calls']['events']==157 and c['accepted']==0 and len(c['attempts'])==5


def test_closed_archive_and_zero_native_entrypoint():
 with tempfile.TemporaryDirectory()as folder:
  root=Path(folder)
  for n in D.FILES:
   p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
  subprocess.run([sys.executable,'-B','-c',"import sys;sys.path[:0]=['tools','src'];import diagnose_clark_explicit_radius as d;assert callable(d.diagnose)"],cwd=root,check=True,capture_output=True)
 import launch_clark_explicit_radius as L
 assert L.OUT==D.OUT.parent
 s=Path(D.__file__).read_text();assert 'A.augmented('not in s and 'paired_family('not in s
