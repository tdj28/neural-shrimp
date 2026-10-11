import copy,json,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import review_fresh_quarter as D

def prior():return dict(passed=True,native_calls=0,parameter_endpoints=['-17/1024','-1/64'],complement_dimension=11)
def rows():return [dict(passed=True,cell=c,complement_dimension=11)for c in D.cells()]


def test_exact17_closed_union_and_every_required_cell():
 assert D.closed_union(rows(),prior())['parameter_endpoints']==['-49/2048','-17/2048']
 for mutation in ('missing','duplicate','width','failed','prior'):
  r=rows();p=prior()
  if mutation=='missing':r.pop()
  elif mutation=='duplicate':r[4]=r[3]
  elif mutation=='width':r[4]['cell']['halfwidth']='1/1024'
  elif mutation=='failed':r[4]['passed']=False
  else:p['passed']=False
  with pytest.raises(ValueError):D.closed_union(r,p)


def test_every_actual_native_pair_and_cell_card_is_required(tmp_path):
 (tmp_path/'native').mkdir()
 for i in range(1,145):
  for kind in('input','return'):(tmp_path/'native'/f'{i:03d}-{kind}.json'if i<=72 else tmp_path/'native'/f'{i:04d}-{kind}.json').write_text('{}')
 for cell in D.cells():
  r=tmp_path/f"cells/{cell['index']:03d}";(r/'native').mkdir(parents=True);start=144+126*(cell['index']-1)
  for i in range(start+1,start+127):
   for kind in('input','return'):(r/'native'/f'{i:04d}-{kind}.json').write_text('{}')
  (r/'cell-input.json').write_text(json.dumps(cell));(r/'cell-outcome.json').write_text(json.dumps(dict(index=cell['index'],endpoints=cell['endpoints'],passed=True,dimension=705,complement_dimension=11,native_calls=126)))
 D.native_histories(tmp_path);p=tmp_path/'cells/016/native/2160-return.json';p.unlink()
 with pytest.raises(ValueError):D.native_histories(tmp_path)


def test_last36_full705_rebuild_and_gradient_seed_corruption(tmp_path,monkeypatch):
 import probe_native705_fresh_center as P
 class Linear:
  n=13
  def cusp_step(self,x,a,b):return[z*arb('0.5')for z in x]
 with ctx.workprec(480):
  monkeypatch.setattr(P,'OUT',tmp_path);w=P.Work();w.calls=216;x=[arb(0)]*705;x[-3:]=[arb(1),arb(1),arb((1,-6))];x[13]=arb(1);x[27]=arb(1);normal=[arb(0)]*705;normal[-3]=arb(1);origin=[arb(0)]*705;spec=D.cells()[0];parameter=D.R.exact(D.F(spec['center']))
  expected=w.residual(Linear(),x,normal,origin,parameter,'target2')
  for p in list((tmp_path/'native').glob('*.json')):p.rename(p.with_name('0'+p.name))
  _,value=D.residual(tmp_path,spec,normal,origin);assert all(value[i,0].overlaps(expected[i,0])for i in range(705))
  p=tmp_path/'native/0217-input.json';d=D.read(p);d['inputs'][0]['g'][0]=D.ball_json(arb(0));p.write_text(json.dumps(d))
  with pytest.raises(ValueError):D.residual(tmp_path,spec,normal,origin)


def test_failure_receipt_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=252*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',dict(x=1))
 w.save('outcome.json',dict(passed=False),True);assert D.read(tmp_path/'outcome.json')['passed']is False


def test_isolated_archive_closure(tmp_path):
 packet=Path(__file__).resolve().parents[2]
 for n in D.FILES:
  source=D.A.ROOT/n
  if not source.is_file():source=packet/n
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes())
 code="import sys,pathlib;sys.path[:0]=['tools/reviews','tools','src'];import review_fresh_quarter as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
