import copy,importlib.util,sys
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[2]/'tools/reviews/review_s2_newgauge_chain.py'
s=importlib.util.spec_from_file_location('newgauge',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def pair():
 p={'normal':['1'],'origin':['0'],'endpoint_root_enclosures':[[0],[1]],'exact_parameter_endpoints':['-1/128','1/128'],'parameter_center':'0','settings':{'halfwidth_exponent':-7}}
 c=copy.deepcopy(p);c.update(anchor={'state_box':[1],'exact_parameter':'1/128'},exact_parameter_endpoints=['1/128','3/128'],parameter_center='1/64');return p,c
def test_exact_transition():M.validate_transition(*pair())
@pytest.mark.parametrize('kind',['phase','gauge','gap','center'])
def test_broken_transition(kind):
 p,c=pair()
 if kind=='phase':c['anchor']['state_box']=[2]
 elif kind=='gauge':c['normal']=['-1']
 elif kind=='gap':c['exact_parameter_endpoints'][0]='2/128'
 else:c['parameter_center']='1/32'
 with pytest.raises(ValueError):M.validate_transition(p,c)
