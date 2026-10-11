import copy,json,os,resource,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_tiny_lm_pair_blocks as W


def test_complete_metadata_only_and_all_upstream_costs(monkeypatch):
 class Store:
  def json(self,n,v,*a,**kw):
   if n=='parity-binding.json':self.bound=v
 class Calls:
  def check(self):pass
 def forbidden(*a,**kw):raise AssertionError('metadata cannot open scientific arrays')
 monkeypatch.setattr(W.np,'load',forbidden);s=Store()
 _,_,rows,worst,_,_,old=W.metadata(s,Calls());assert len(rows)==1308 and len(old)==224
 costs=s.bound['all_upstream_hvp_costs'];assert [len(costs[k])for k in ('response_cold','response_actual','two_snapshot_actual','ordering_actual','parity_successful_cold','parity_actual')]==[4,256,240,160,2,224]
 assert worst==max(x for v in costs.values()for x in v)


def test_card_roles_and_literal_independent_authority():
 c=json.loads((ROOT/W.CARD).read_text());c['status']='source_prepared'
 with pytest.raises(ValueError):W.card_check(c,'profile')
 c['status']='profile_admitted';c['source_review']='synthetic';W.card_check(c,'profile')
 bad=copy.deepcopy(c);bad['parity_authority']['worker']='0'*64
 with pytest.raises(ValueError):W.card_check(bad,'profile')
 with pytest.raises(ValueError):W.card_check(c,'target')
 c['settings']['target_counts']['hvp']=127
 with pytest.raises(ValueError):W.card_check(c,'profile')


def test_call_retention_and_dedup_closing_reserve(tmp_path):
 s=W.Store(tmp_path/'run');c=W.Calls(s,'target');u=np.ones(15776);c.label='synthetic'
 x=c.call('hvp',lambda:np.full(15776,np.nan),u,u);assert not np.isfinite(x).all()
 assert c.attempted==c.returned=={'hvp':1}and len(list((s.root/'events').glob('*.json')))==2
 before=s.used;s.alias('same',np.repeat(u[None],17,axis=0));assert s.used-before<20000 and len(s.objects)==2
 s.used=W.CAP-W.RESERVE
 with pytest.raises(ValueError):s.json('blocked',{})
 s.json('failure',{'passed':False},True)


def test_saved_state_direction_recurrence_guards(monkeypatch):
 u=np.ones(15776);states=np.arange(17.)[:,None]*u;trace=np.repeat(u[None],17,axis=0);changed=[False];records=[dict(case='test/P',state=i,direction=20+i,returned=40+i)for i in range(16)]
 monkeypatch.setattr(W.S,'alias',lambda *a:trace)
 monkeypatch.setattr(W.S,'read',lambda *a:dict(indices=list(range(16))))
 def obj(run,p,files):
  if p<20:return states[p]+(.1 if changed[0]else 0)
  if p<40:return u
  return np.zeros(15776)
 monkeypatch.setattr(W.S,'load_object',obj);case=dict(direction=u,reference=states)
 assert np.array_equal(W.replay_saved_order(Path('/tmp'),{},records,'test','P',case,.2),trace)
 changed[0]=True
 with pytest.raises(ValueError,match='recurrence'):W.replay_saved_order(Path('/tmp'),{},records,'test','P',case,.2)


def test_all_four_qualifications_before_original_reports():
 rows=[dict(seed=s,rate_role=r,qualified=True,comparison=dict(even_relative_ordered=2.,even_relative_parity_even=.01))for s in(28161,28162)for r in('original','quarter_step')]
 assert all(W.H.descriptive(rows).values());rows[1]['qualified']=False;assert not any(W.H.descriptive(rows).values())


def test_isolated_archive_and_actual_soft_limit_entrypoint(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_tiny_lm_pair_blocks as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=env);assert p.returncode==0,p.stderr
 def inherited():resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,8*1024**3))
 p=subprocess.run([sys.executable,'-B',str(tmp_path/'tools/explore_tiny_lm_pair_blocks.py'),'--help'],cwd=tmp_path,capture_output=True,text=True,env=env,preexec_fn=inherited);assert p.returncode==0,p.stderr


def test_counts_and_external_caps():
 s=(ROOT/'tools/launch_tiny_lm_pair_blocks.py').read_text();assert 'else 255'in s and '194*1024**2'in s and 'attempt001-profile'in s
 assert W.SETTINGS['target_counts']=={'hvp':128}and W.CAP==192*1024**2 and W.H.VECTOR_BOUND==356
 assert W.Calls.__init__.__code__.co_consts.count(240)==1
