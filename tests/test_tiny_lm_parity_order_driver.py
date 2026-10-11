import copy,json,os,resource,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_tiny_lm_parity_order as W


def test_complete_metadata_only_and_cost_floor(monkeypatch):
 class Store:
  def json(self,*a,**kw):pass
 class Calls:
  def check(self):pass
 def forbidden(*a,**kw):raise AssertionError('metadata cannot open scientific arrays')
 monkeypatch.setattr(W.np,'load',forbidden)
 _,_,rows,worst=W.metadata(Store(),Calls());assert len(rows)==1308 and worst>=max(r['seconds']for r in rows if r['operation']=='hvp')


def test_card_roles_and_separate_admission():
 c=json.loads((ROOT/W.CARD).read_text())
 with pytest.raises(ValueError):W.card_check(c,'profile')
 c['status']='profile_admitted';c['source_review']='synthetic';W.card_check(c,'profile')
 with pytest.raises(ValueError):W.card_check(c,'target')
 c['settings']['target_counts']['hvp']=223
 with pytest.raises(ValueError):W.card_check(c,'profile')


def test_call_retention_on_nonfinite_and_deduplicated_trajectories(tmp_path):
 s=W.Store(tmp_path/'run');c=W.Calls(s,'target');u=np.ones(15776);c.label='synthetic'
 x=c.call('hvp',lambda:np.full(15776,np.nan),u,u);assert not np.isfinite(x).all()
 assert c.attempted==c.returned=={'hvp':1}and len(list((s.root/'events').glob('*.json')))==2
 before=s.used;s.alias('same',np.repeat(u[None],17,axis=0));assert s.used-before<20000
 assert len(s.objects)==2
 s.used=W.CAP-W.RESERVE
 with pytest.raises(ValueError):s.json('blocked',{})
 s.json('failure',{'passed':False},True)


def test_profile_target_constants_and_launcher():
 s=(ROOT/'tools/launch_tiny_lm_parity_order.py').read_text();assert 'else 260'in s and '194*1024**2'in s and 'attempt001-profile'in s
 assert W.SETTINGS['target_counts']=={'hvp':224}and W.CAP==192*1024**2


def test_isolated_archive_and_real_inherited_import(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_tiny_lm_parity_order as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=env);assert p.returncode==0,p.stderr
 def inherited():resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,8*1024**3))
 p=subprocess.run([sys.executable,'-B',str(tmp_path/'tools/explore_tiny_lm_parity_order.py'),'--help'],cwd=tmp_path,capture_output=True,text=True,env=env,preexec_fn=inherited);assert p.returncode==0,p.stderr


def test_literal_reused_reverse_states_and_actions(monkeypatch):
 u=np.zeros(15776);u[0]=1.;states=np.arange(17,dtype=float)[:,None]*np.ones((1,15776));trace=np.repeat(u[None],17,axis=0);changed=[False]
 def read(p):
  if p.name=='state-order.json':return dict(indices=list(W.H.R),operation_first=61,operation_last=76)
  return dict(eta=.2,seed=28161)
 def load(p,**kw):
  if p.name=='reverse-trace.npy':return trace
  index=int(p.name.split('-')[1])-61
  if p.name.endswith('-theta.npy'):return states[15-index]+(.1 if changed[0]else 0)
  if p.name.endswith('-vector.npy'):return trace[index]
  if p.name.endswith('-return.npy'):return np.zeros(15776)
  raise AssertionError(p)
 monkeypatch.setattr(W.S,'read',read);monkeypatch.setattr(W.np,'load',load)
 c=dict(direction=u,reference=states);assert np.array_equal(W.original_reverse(28161,c,.2),u)
 changed[0]=True
 with pytest.raises(ValueError,match='state/vector/action'):W.original_reverse(28161,c,.2)
