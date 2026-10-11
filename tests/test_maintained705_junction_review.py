import ast,copy,sys,json,os,subprocess
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'src')]
import review_maintained705_junction as A


def fake_card():
 c=A.read(ROOT/A.CARD);c['status']='reviewed';c['anchor']={k:'pending'for k in ('root','source','outer','worker','endpoint','proof')};return c


def test_complete_all20_history_and_actual_rotations():
 c=fake_card();assert len(A.selection(c))==20
 for field,value in [('selected_attempts',list(range(1,19))),('failed_attempts',[]),('anchor',None),('status','prepared')]:
  bad=copy.deepcopy(c);bad[field]=value
  with pytest.raises(ValueError):A.selection(bad)
 bad=copy.deepcopy(c);bad['producer']['workers'][9]['joined']=True
 with pytest.raises(ValueError):A.selection(bad)
 bad=copy.deepcopy(c);bad['producer']['workers'][10]['settings']['halfwidth_exponent']=-3
 with pytest.raises(ValueError):A.selection(bad)


def test_entire705_junction_and_projected_scalar_required():
 with ctx.workprec(480):
  n=705;z=[arb(0)]*n;normal=z.copy();normal[0]=arb(1);r=[arb(1)]*n;data=(z,z,z,z,normal,z,r,arb(1),arb(1),arb(0,1),r,z,z);pc=dict(settings=dict(parameter='0'));box=A.T.vector_json(z)
  ans=A.membership(data,pc,box);assert ans['passed']and ans['coordinates']==705
  for idx in (0,13,702,703,704):
   bad=copy.deepcopy(box);bad[idx]=A.T.ball_json(arb(2))
   with pytest.raises(ValueError):A.membership(data,pc,bad)
  boundary=copy.deepcopy(box);boundary[20]=A.T.ball_json(arb(1))
  with pytest.raises(ValueError):A.membership(data,pc,boundary)


def test_original_reducer_called_unchanged_and_owned_caps():
 text=(ROOT/'tools/reviews/review_maintained705_junction.py').read_text();tree=ast.parse(text);calls={ast.unparse(v.func)for v in ast.walk(tree)if isinstance(v,ast.Call)}
 assert 'T.tube'in calls and 'H.cubic_membership'in calls and not any('cusp_step'in x for x in calls)
 launcher=(ROOT/'tools/launch_maintained705_junction_review.py').read_text();assert "'internal_seconds':7200"in launcher and "'external_seconds':7215"in launcher and 'attempt001'in launcher


def test_archived_source_closure(tmp_path):
 for n in A.FILES:
  f=tmp_path/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','src'];import review_maintained705_junction as A;loaded={str(Path(m.__file__).resolve().relative_to(Path.cwd()))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(Path.cwd())};assert loaded<=set(A.FILES),loaded-set(A.FILES)"
 out=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True);assert out.returncode==0,out.stderr


def test_known_junction_and_curve_equation_identity():
 model=dict(network=dict(width=3,ridge='1/1024'),x=[['1']])
 native={'native.py':dict(sha256='same',bytes=1)}
 lift=dict(native_source=native);curve=dict(model=model,native_source=native)
 A.same_equations(lift,curve,model)
 with pytest.raises(ValueError):A.same_equations(lift,curve,dict(network=dict(width=4)))
 with pytest.raises(ValueError):A.same_equations(dict(native_source={'native.py':dict(sha256='different',bytes=1)}),curve,model)
