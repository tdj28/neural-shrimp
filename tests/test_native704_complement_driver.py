import importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native704_complement as D

def domain():
 b=D.ball_json
 return {'center':[b(arb(i))for i in range(704)],'whole':[b(arb(i,arb('1/8')))for i in range(704)],'tangent':[b(arb(i+1,arb('1/16')))for i in range(704)]}

def acquired(dom,j,kind):
 v=dom['center'if kind=='center'else'whole'];t=dom['tangent'];off=39*j
 return dict(phase=j,kind=kind,state=v[off:off+13],rates=v[-2:],state_tangent=[D.ball_json(arb(0))]*13 if kind=='center'else t[off:off+13],rate_tangent=[D.ball_json(arb(0))]*2 if kind=='center'else t[-2:])

def test_both_actual_rates_and_true_tangents():
 d=domain()
 for k in ['center','whole_tangent']:
  for j in [0,17]:D.acquired_input(acquired(d,j,k),d,k,j)
 for field in ['state','rates','state_tangent','rate_tangent']:
  a=acquired(d,17,'whole_tangent');a[field]=list(a[field]);a[field][-1]=D.ball_json(arb(-99))
  with pytest.raises(ValueError):D.acquired_input(a,d,'whole_tangent',17)
 a=acquired(d,0,'center');a['rate_tangent']=[D.ball_json(arb(1))]*2
 with pytest.raises(ValueError):D.acquired_input(a,d,'center',0)

def test_exact_domain_without_outward_enlargement():
 with ctx.workprec(480):
  a=[D.ball_json(arb(0,arb('1/8')))];D.exact_cover(a,[arb(0,arb('1/16'))],'fixture')
  with pytest.raises(ValueError):D.exact_cover(a,[arb(0,arb('1/4'))],'outside')
  with pytest.raises(ValueError):D.exact_cover(a,[],'missing')

def test_closing_reserve(tmp_path):
 used=[62*1024**2-10]
 with pytest.raises(ValueError):D.save_bound(tmp_path,'large.json',{'large':'a'*100},used)
 D.save_bound(tmp_path,'failure.json',{'passed':False,'error':'retained'},used,True)
 D.save_bound(tmp_path,'manifest.json',{'files':{'failure.json':D.entry(tmp_path/'failure.json')}},used,True)
 assert json.loads((tmp_path/'failure.json').read_text())['passed']is False

def test_archive_import_closure(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code='import sys;sys.path.insert(0,'+repr(str(tmp_path/'tools/reviews'))+');import review_native704_complement as d;assert d.ROOT==__import__("pathlib").Path('+repr(str(tmp_path))+');assert all((d.ROOT/n).is_file() for n in d.FILES)'
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_launcher_matches_fixed_worker():
 p=ROOT/'tools/reviews/launch_native704_complement_review.py';s=p.read_text()
 assert 'tools/reviews/review_native704_complement.py'in s and "'internal_seconds':120"in s and "'external_seconds':135"in s
 assert str(D.OUT.parent)in s and 'probe_native_unit_pair_correlated'not in s

def test_small_full_arithmetic_profile():
 with ctx.workprec(480):assert D.synthetic(8,lambda:None)['passed']
