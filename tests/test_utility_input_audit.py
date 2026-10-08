import importlib.util
from pathlib import Path
import hashlib
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'tools/audit_utility_inputs.py'
s=importlib.util.spec_from_file_location('audit_utility_inputs',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture():
 n=1003;state=np.arange(n*2*3,dtype=float).reshape(n,2,3);loss=np.arange(n*2*2,dtype=float).reshape(n,2,2)+1
 alpha=np.full((n,2),-.1);alpha[1000:]=0
 trace={'state':state,'loss':loss,'alpha':alpha,'rates':np.ones((n,2,2)),'distance':np.zeros((n,2)),'initial_state':np.zeros((2,3))}
 ref={'h':0.,'o':0.,'alpha':-.1,'orbit':np.zeros((2,3)),'closure_max':0.,'primitive_separation':1.,'loss_mean':1.,'loss_max':2.}
 return trace,[ref,{**ref,'alpha':0.}]

def test_fork_clock_and_loss_history():
 t,r=fixture();f=m.local_fork(t,r,1,0)
 assert f['array_index']==[1000,1] and f['accepted_updates_at_fork']==2002
 assert f['prefix_pair_indices_inclusive']==[1,1000]
 assert f['fork_training_loss']==t['loss'][1001,1,0]
 assert f['prefix_mean_training_loss']==t['loss'][1:1001,1].mean()
 assert f['state']['sha256']==hashlib.sha256(t['state'][1000,1].astype('<f8').tobytes()).hexdigest()

def test_recorded_zero_not_rounded():
 t,r=fixture();t['alpha'][1000]=-1e-18;t['alpha'][1001:]=0
 assert m.local_fork(t,r,0,0)['array_index'][0]==1001

def test_short_or_nonfinite_history_rejected():
 t,r=fixture();t['alpha'][500]=0
 with pytest.raises(ValueError,match='history'):m.local_fork(t,r,0,0)
 t,r=fixture();t['loss'][300,0,0]=np.nan
 with pytest.raises(ValueError,match='finite'):m.local_fork(t,r,0,0)

@pytest.mark.parametrize('a,b,want',[(1,2,True),(2,1,True),(1,2.0001,False),(0,1,False),(np.nan,1,False),(np.inf,1,False)])
def test_support(a,b,want):assert m.within_support(a,b)==want

def test_nomination_deterministic_without_pulse_data():
 forks=[{'point':p,'prefix_mean_training_loss':1.,'control_recovery_uniform_factor':.99} for p in (0,1) for _ in (0,1)]
 rows=[{'id':i,'class':c,'distance_pass':True,'tail_mean_training_loss':1.,'log_rates':[ord(i),0.]} for i,c in [('a','periodic_candidate'),('b','periodic_candidate'),('c','chaotic_candidate'),('d','chaotic_candidate')]]
 a=m.nominate(rows,forks);b=m.nominate(list(reversed(rows)),forks)
 assert a==b and [x['candidate'] for x in a]==['a','c','b','d']
 for r in rows:r['pulse_outcome']=np.nan
 assert m.nominate(rows,forks)==a

def test_no_support_missing_or_same_candidate_reuse():
 forks=[{'point':p,'prefix_mean_training_loss':1.,'control_recovery_uniform_factor':.99} for p in (0,1)]
 rows=[{'id':'a','class':'periodic_candidate','distance_pass':False,'tail_mean_training_loss':1.,'log_rates':[1.,0.]}]
 assert all(x['candidate'] is None for x in m.nominate(rows,forks))

def test_signed_zero_bits_retained():
 a=m.array_identity(np.array([0.,-0.]))
 assert a['binary64_hex']==['0000000000000000','8000000000000000'] and a['shape']==[2]
