import copy,json,os,resource,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_tiny_lm_block_mean as W


def test_complete_metadata_only_and_all_upstream_costs(monkeypatch):
 class Store:
  def json(self,n,v,*a,**kw):
   if n=='pair-binding.json':self.bound=v
 class Calls:
  def check(self):pass
 def forbidden(*a,**kw):raise AssertionError('metadata cannot open scientific arrays')
 monkeypatch.setattr(W.np,'load',forbidden);store=Store();_,_,rows,worst,_,_,old=W.metadata(store,Calls());assert len(rows)==1308 and len(old)==224 and len(store.bound['all130_costs'])==130
 assert worst>=max(store.bound['all130_costs'])


def test_card_roles_and_literal_independent_authority():
 c=json.loads((ROOT/W.CARD).read_text());c['status']='source_prepared'
 with pytest.raises(ValueError):W.card_check(c,'profile')
 c['status']='profile_admitted';c['source_review']='synthetic';W.card_check(c,'profile')
 bad=copy.deepcopy(c);bad['pair_authority']['worker']='0'*64
 with pytest.raises(ValueError):W.card_check(bad,'profile')
 with pytest.raises(ValueError):W.card_check(c,'target')
 c['settings']['target_counts']['hvp']=511
 with pytest.raises(ValueError):W.card_check(c,'profile')


def test_call_retention_and_dedup_closing_reserve(tmp_path):
 s=W.Store(tmp_path/'run');c=W.Calls(s,'target');u=np.ones(15776);c.label='synthetic'
 x=c.call('hvp',lambda:np.full(15776,np.nan),u,u);assert not np.isfinite(x).all()
 assert c.attempted==c.returned=={'hvp':1}and len(list((s.root/'events').glob('*.json')))==2
 before=s.used;s.alias('same',np.repeat(u[None],17,axis=0));assert s.used-before<20000 and len(s.objects)==2
 s.used=W.CAP-W.RESERVE
 with pytest.raises(ValueError):s.json('blocked',{})
 s.json('failure',{'passed':False},True)


def test_isolated_archive_and_actual_soft_limit_entrypoint(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_tiny_lm_block_mean as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=env);assert p.returncode==0,p.stderr
 def inherited():resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,8*1024**3))
 p=subprocess.run([sys.executable,'-B',str(tmp_path/'tools/explore_tiny_lm_block_mean.py'),'--help'],cwd=tmp_path,capture_output=True,text=True,env=env,preexec_fn=inherited);assert p.returncode==0,p.stderr


def test_counts_and_external_caps():
 s=(ROOT/'tools/launch_tiny_lm_block_mean.py').read_text();assert 'else 915'in s and '514*1024**2'in s and 'attempt001-profile'in s
 assert W.SETTINGS['target_counts']=={'hvp':512}and W.CAP==512*1024**2 and W.H.VECTOR_BOUND==1280
 assert W.Calls.__init__.__code__.co_consts.count(900)==1

def test_actual_four_saved_baselines_and_inherited_nonlinear_responses(monkeypatch):
 class Store:
  def json(self,*a,**kw):pass
 class Calls:
  def check(self):pass
 def forbidden(*a,**kw):raise AssertionError('saved baseline control cannot construct a native model')
 monkeypatch.setattr(W.N,'LMProblem',forbidden)
 run,files,records,*_=W.metadata(Store(),Calls());parent,_=W.N.PARENTS['target']
 for seed in(28161,28162):
  theta,x,y,eta,binding=W.N.select(parent,seed,600)
  for role in('original','quarter_step'):
   case,rate,b=W.P.load_case(run,files,records,seed,role,theta,eta)
   old,responses,slopes,eps=W.comparison_inputs(run,files,f'seed-{seed}-{role}',case,role)
   assert len(responses)==2 and all(v.shape==(15776,)for v in responses)
   assert W.H.compare(old['ordered'],old,responses,slopes)['qualified']
