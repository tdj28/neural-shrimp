import pytest
from neural_shrimp.cubic_continuation_policy import next_step

def step(h=-6,s=0,**kw):return next_step(h,s,**dict(root_passed=True,joined=True,other_gates=True,contraction='1/10',**kw))
def test_two_strong_passes_expand_only_within_cap():
 a=step();assert a==dict(action='advance',halfwidth=-6,strong=1,rotate=False)
 assert step(-6,1)['halfwidth']==-5 and step(-3,1)['halfwidth']==-3

def test_only_completed_root_bound_failure_halves():
 assert next_step(-6,1,root_passed=False,joined=False,other_gates=True,contraction=None)['action']=='halve'
 assert next_step(-6,1,root_passed=True,joined=False,other_gates=True,contraction='1/10')['action']=='stop'
 assert next_step(-6,1,root_passed=False,joined=False,other_gates=False,contraction=None)['action']=='stop'
 assert next_step(-9,0,root_passed=False,joined=False,other_gates=True,contraction=None)['action']=='stop'

def test_gauge_rotation_requires_proved_conditioning_trigger():
 assert next_step(-6,1,root_passed=True,joined=True,other_gates=True,contraction='1/4')==dict(action='advance',halfwidth=-6,strong=0,rotate=True)
 with pytest.raises(ValueError):next_step(-2,0,root_passed=True,joined=True,other_gates=True,contraction='1/10')
