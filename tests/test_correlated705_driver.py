import sys,json,gzip,os,shutil,subprocess
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_correlated705 as W
from neural_shrimp.rigorous_orbits import vector_json


def geometry():
 with ctx.workprec(256):
  c=[arb(0)]*702+[arb(1)/16,arb(1)/32,arb(1)/64];s=[arb(1)/1024]*705;a=[arb(1)/4096]*705;weights=['1']*705;error=[arb(0,arb(1)/1024)]*705;d=arb(0,arb(1)/32)
  proposal=dict(center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),weights=weights,normal=['1']+['0']*704,origin=['0']*705,parameter_center='15/2048',settings=dict(halfwidth_exponent=-7,radius_exponent=-10))
  domain=dict(center=proposal['center'],slope=proposal['slope'],curvature=proposal['curvature'],error=vector_json(error),whole_box=vector_json([c[i]+s[i]*d+a[i]*d*d/2+error[i]for i in range(705)]),parameter='15/2048',halfwidth_exponent=-5,radius_exponent=-10)
  return proposal,domain


def test_all705_actual_correction_domains_and_changed_width_rejected():
 p,d=geometry()
 with ctx.workprec(480):
  v=W.actual_geometry(p,d);assert len(v[-1])==705
  d['error'][7]=vector_json([arb(0)])[0]
  with pytest.raises(ValueError):W.actual_geometry(p,d)
 p,d=geometry();d['halfwidth_exponent']=-4
 with pytest.raises(ValueError):W.actual_geometry(p,d)


def test_streamed_dyadic_matrix_and_interrupted_bytes_retained(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();a=arb_mat([[arb(1)/3,arb(0)],[arb(0),arb(2)]])
 w.matrix('a.json.gz',a);r=json.loads(gzip.decompress((tmp_path/'a.json.gz').read_bytes()));assert r['encoding']=='mid-rad-dyadic-pairs'and len(r['rows'])==2
 assert w.used==(tmp_path/'a.json.gz').stat().st_size
 w.stopped=True
 with pytest.raises(ValueError):w.matrix('partial.json.gz',a)
 assert w.used==sum(p.stat().st_size for p in tmp_path.iterdir())
 w.save('outcome.json',dict(passed=False,error='retained interrupted matrix'),True)


def test_final_reserve_and_literal_launcher_caps(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=508*1024**2-1
 with pytest.raises(ValueError):w.save('large.json',{'x':'y'*20})
 w.save('outcome.json',dict(passed=False),True);assert (tmp_path/'outcome.json').is_file()
 s=(ROOT/'tools/launch_correlated705_review.py').read_text();assert 'tools/reviews/review_correlated705.py'in s and "'internal_seconds':240,'external_seconds':255"in s and "'worker_rss_kib':6*1024**2"in s and '514*1024**2'in s and 'parents[1]'in s


def test_actual_gradient_phase_order_and_missing_pair_fail(tmp_path):
 ev=tmp_path/'events';ev.mkdir();a=dict(phase=8,direction='q',d={},inputs=[]);b=dict(phase=0,direction='q',returns=[])
 (ev/'0001-gradient_path_input.json.gz').write_bytes(gzip.compress(json.dumps(a).encode()));(ev/'0002-gradient_path_return.json.gz').write_bytes(gzip.compress(json.dumps(b).encode()))
 with pytest.raises(ValueError):W.gradients(tmp_path,0)
 with pytest.raises(ValueError):W.gradients(tmp_path,2)


def test_archive_import_closure_in_isolation(tmp_path):
 for n in W.FILES:
  dest=tmp_path/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
 code="""import sys
from pathlib import Path
import review_correlated705 as W
root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.FILES),(loaded-set(W.FILES))
assert Path(W.__file__).resolve().is_relative_to(root)
"""
 env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools/reviews'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
