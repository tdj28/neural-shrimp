import importlib.util
from pathlib import Path
import copy
import pytest
from flint import ctx

def module():
 p=Path(__file__).parents[1]/'tools/probe_floquet_complement.py';s=importlib.util.spec_from_file_location('floquet_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture():
 flat=list(range(704));J=[[0]*704 for _ in range(704)];rows=[]
 for j in range(18):
  off=39*j
  for d in [1,2]:rows.append({'phase':j,'direction':'q'if d==1 else'r','input':flat[off:off+13]+flat[off+13*d:off+13*(d+1)]+flat[-2:],'block':[[0]*26 for _ in range(26)]})
 proof={'dimension':13,'period':18,'shooting_dimension':704,'parameter_endpoints':['0','1/32768'],'whole_box':flat,'whole_jacobian':J}
 old={'whole_box':flat,'native_blocks':[[[0]*13 for _ in range(13)]for _ in range(18)],'exact_parameter_endpoints':['0','1/32768']}
 return proof,rows,old

def test_literal_domains_and_all_native_jacobians_bound():
 m=module();p,r,o=fixture();assert len(m.admitted_native_blocks(p,r,o)[1])==18
 for mutate in [lambda p,r,o:r[-1].update(phase=16),lambda p,r,o:r[-1]['input'].__setitem__(0,-1),lambda p,r,o:p['whole_jacobian'][39*17].__setitem__(39*17,1),lambda p,r,o:o['native_blocks'][0][0].__setitem__(0,1)]:
  p,r,o=fixture();mutate(p,r,o)
  with pytest.raises(ValueError):m.admitted_native_blocks(p,r,o)

def test_fullsize_unrelated_profile():
 m=module()
 with ctx.workprec(384):
  x=m.profile(lambda:None);assert x['passed']and x['period']==18 and x['complement_dimension']==11 and x['native_calls']==0
