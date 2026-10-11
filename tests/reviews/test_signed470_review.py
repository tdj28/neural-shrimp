import sys,copy
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'src')]
import review_signed470_anchor as A
import review_signed470 as D
from neural_shrimp.rigorous_orbits import ball_json

def fixture():
 b=lambda x:ball_json(arb(x));n=2;p=2;x=[b(0)]*n;q=[b(0),b(1)];r=[b(1),b(0)];w=[b(1),b(0)];rates=[b(1),b(2)];root=x+rates+q+r+[b(.1)]+w;unit=(x+q+r)*p+rates+[b(.1)];records=[]
 for k in range(p):
  B=[[b((-1 if k==0 else 1)if i==j==0 else int(i==j))for j in range(4)]for i in range(4)]
  for name,mode in [('q',q),('r',r)]:records.append(dict(phase=k,direction=name,input=x+mode+rates,value=[[v]for v in x+mode],block=copy.deepcopy(B)))
 return root,unit,records

def test_independent_signed_transport_and_wrong_df():
 with ctx.workprec(512):
  r,u,e=fixture();v=A.negative_lift(r,u,e,n=2,p=2);assert len(v)==10 and A.M.point(v[2])==1 and A.M.point(v[6])==-1
  e[1]['block'][0][0]=ball_json(arb(1))
  with pytest.raises(ValueError,match='direction-independent'):A.negative_lift(r,u,e,n=2,p=2)

def test_same_root_and_mode_conventions():
 r,u,e=fixture();u[-1]=ball_json(arb(.2))
 with pytest.raises(ValueError,match='same entire'):A.negative_lift(r,u,e,n=2,p=2)

def test_full470_synthetic_reduction_and_negative_sign_is_explicit(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();card=D.read(ROOT/D.CARD)
 with ctx.workprec(512):D.profile(w,card)
 assert D.read(tmp_path/'profile.json')['passed'];s=(ROOT/'tools/reviews/review_signed470_locus_math.py').read_text();assert 'expected_sigma=1'in s and "chart['sigma']==expected_sigma"in s

def test_native_negative_closure_block_sign():
 b=lambda x:ball_json(arb(x));box=[b(0)]*10;chart=dict(period=2,dimension=2,sigma=-1,whole_box=box,normal=['0']*8+['1','0']);events=[]
 for k in range(2):events.append(dict(phase=k,input=box[4*k:4*(k+1)]+box[-2:],block=[[b(int(i==j))for j in range(4)]for i in range(4)],rates=[[b(0),b(0)]for i in range(4)]))
 J=D.E.rebuild_whole(chart,events);assert J[4,0]==-1 and J[6,2]==1 and J[7,3]==1

def test_isolated_actual_file_archive_import_closure(tmp_path):
 import shutil,subprocess
 for name in D.FILES:
  target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
 script=tmp_path/'closure.py'
 script.write_text("from pathlib import Path\nimport sys\na=Path(__file__).resolve().parent\nsys.path[:0]=[str(a/'tools/reviews'),str(a/'src')]\nimport review_signed470 as d\nassert d.ROOT==a\nfor name in d.FILES:\n assert (a/name).is_file(),name\nfor m in list(sys.modules.values()):\n p=getattr(m,'__file__',None)\n if p and ('neural_shrimp' in str(p) or '/tools/reviews/' in str(p)):\n  assert Path(p).resolve().is_relative_to(a),p\n")
 subprocess.run([sys.executable,'-I',str(script)],cwd=tmp_path,check=True,capture_output=True,text=True)
