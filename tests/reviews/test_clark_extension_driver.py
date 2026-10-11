import copy,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews')]
import review_clark_extension as D

def test_actual_complete_event_metadata_and_exact_endpoint_inputs(tmp_path,monkeypatch):
 monkeypatch.setattr(D.D,'OUT',tmp_path);w=D.Work()
 with ctx.workprec(512):paths,kinds,known,ib=D.metadata(w)
 assert len(paths)==207 and len(known)==51 and sum(D.EXPECTED[k]for k in ('native_even_odd_phase','clark_gradient_path_return','paired_native_phase'))==93
 root=D.read(D.PARENT/'run/endpoint-root.json');c=D.read(paths[175]);whole=D.read(paths[180]);gain=D.read(D.PARENT/'run/outcome.json')['fixed_gain'];D.endpoint_inputs(root,c,whole,gain)
 bad=copy.deepcopy(whole);bad['input'][0]=bad['input'][1]
 with pytest.raises(ValueError,match='acquired domains'):D.endpoint_inputs(root,c,bad,gain)
 implicit=D.read(paths[205]);solve=D.read(paths[201]);assert implicit['input']==whole['input'][:26]and implicit['squared_amplitude']==whole['squared_amplitude']and implicit['paired_tangent']==solve['solution']

def test_closed_interval_root_join_and_phase_displacement():
 def enc(x):return D.serial(arb(x))
 t=dict(center=[enc(0)]*51,slope=[enc(1)]*51,curvature=[enc(0)]*51,radii=[enc(1)]*51,parameter=enc(0),halfwidth=enc(1))
 b=arb('1/2','1/8');box=[b]*51;assert D.interval_join(box,b,t)['passed'];box[30]+=3;assert not D.interval_join(box,b,t)['passed']

def test_archive_closure_no_producer_math_and_owned_bounds(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 result=subprocess.run([sys.executable,'-c',"import sys,runpy;sys.path.insert(0,'tools/reviews');runpy.run_path('tools/reviews/review_clark_extension.py',run_name='import_only');assert not any('rigorous_clark_'in n for n in sys.modules)"],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15);assert result.returncode==0,result.stderr
 t=(ROOT/'tools/launch_clark_extension_review.py').read_text();assert 'clark-correlated-extension-independent/attempt001'in t and "'internal_seconds':180"in t and "'external_seconds':195"in t and '130*1024**2'in t
