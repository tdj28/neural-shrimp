import copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import explore_cifar_curved512 as W
import launch_cifar_curved512 as L
W.B.torch.set_num_threads(1);W.B.torch.set_num_interop_threads(1)

def card():return dict(W.read(ROOT/W.CARD),status='reviewed')

def test_fixed_first_curved_point_and_fresh_operator_only():
 c=card();W.check_card(c);assert W.BASE==L.OUT
 for key,value in [('selected_alpha',1/64),('selected_index',1),('old_basis_reused',512),('origin','final-point.npy'),('gradient','initial-gradient.npy')]:
  bad=copy.deepcopy(c);bad[key]=value
  with pytest.raises(ValueError):W.check_card(bad)


def test_complete_actual_metadata_exclusive_saves_without_arrays(tmp_path,monkeypatch):
 W.install();store=W.Store(tmp_path/'run',limit=8*1024**2);budget=W.Budget(store.event)
 def no(*a,**k):raise AssertionError('scientific arrays/model forbidden before profile')
 monkeypatch.setattr(W.np,'load',no);monkeypatch.setattr(W.B,'CountedProblem',no)
 parent,ops,old,events=W.bind(card(),{n:W.entry(ROOT/n)["sha256"]for n in W.B.SOURCES},W.B.environment(),budget,store)
 assert parent==Path(card()['producer']['root'])/'run'and len(ops)==3
 assert old['comparison']['chosen_alpha']==1/32
 assert max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='hvp')>=1.1099920831620693
 assert (store.root/'ancestry-input-binding.json').is_file()and(store.root/'curved-endpoint-binding.json').is_file()
 assert budget.attempted==dict.fromkeys(W.CAPS,0)


def test_forecast_includes_full_cold_operator_cost():
 f=W.forecasts(.05,3.,dict(hvp=1.1099920831620693,direct_gradient=.02),10.,1024**2,1024**2)
 assert f['forecast_seconds']>2*514*1.1099920831620693 and f['forecast_raw_bytes']>900*1024**2
 assert not W.forecasts(2.,3.,dict(hvp=1.1099920831620693,direct_gradient=.02),10.,1024**2,1024**2)['passed']


def test_native_returns_are_retained_before_budget_stop(tmp_path):
 store=W.Store(tmp_path/'run',limit=8*1024**2);b=W.Budget(store.event)
 class Problem:
  def hvp(self,x,v):
   def action():b.stopped=True;return W.B.torch.ones_like(v)
   return b.call('hvp',action)
 with pytest.raises(ValueError):W.E.Native(Problem(),store,b).hvp(np.zeros(98666),np.ones(98666))
 rows=[json.loads(s)for s in(store.root/'events.jsonl').read_text().splitlines()]
 assert [v['kind']for v in rows]==['native_input','operation_started','operation_returned','native_output']
 assert(store.root/rows[-1]['result']['file']).is_file()and b.attempted==b.returned


def test_isolated_archive_and_inherited_soft_limit(tmp_path):
 W.install()
 for n in W.B.SOURCES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import explore_cifar_curved512 as w;w.install();r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.B.SOURCES),a-set(w.B.SOURCES)"
 out=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert out.returncode==0,out.stderr
