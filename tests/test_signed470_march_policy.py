import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fractions import Fraction as F
import pytest
from neural_shrimp.signed470_march_policy import next_state,next_parameter

def test_two_strong_passes_grow_only_with_entire_root_join():
 a=next_state(-3,0,uniform=True,primitive=True,joined=True,contraction=F(1,10));assert a['width']==-3 and a['strong']==1
 b=next_state(a['width'],a['strong'],uniform=True,primitive=True,joined=True,contraction=F(1,10));assert b['width']==-2 and b['action']=='advance'
 assert next_state(-1,1,uniform=True,primitive=True,joined=True,contraction=F(1,10))['width']==-1
 assert next_state(-3,1,uniform=True,primitive=True,joined=True,contraction=F(1,5))['width']==-3

def test_only_complete_joined_bound_failure_halves():
 kw=dict(uniform=False,primitive=True,joined=True,contraction=F(2))
 assert next_state(-3,1,**kw)['width']==-4
 assert next_state(-8,0,**kw)['action']=='stop'
 for changes in [dict(primitive=False),dict(joined=False),dict(uniform=True,joined=False)]:
  assert next_state(-3,0,**dict(kw,**changes))['action']=='stop'

def test_fixed_exact_endpoint_schedule():
 assert next_parameter('-81/256',-3)==('-113/256','-145/256','-81/256')
 assert next_parameter('-145/256',-2)==('-209/256','-273/256','-145/256')
 with pytest.raises(ValueError):next_parameter('0',0)
