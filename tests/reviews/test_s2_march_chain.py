import copy,importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('chain',Path(__file__).parents[2]/'tools/reviews/review_s2_march_chain.py');D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
def sample():
 prev={'normal':['1'],'origin':['0'],'endpoint_root_enclosures':[[0],[1]],'exact_parameter_endpoints':['-1/64','1/64'],'parameter_center':'0'}
 c={'normal':['1'],'origin':['0'],'anchor':{'state_box':[1],'exact_parameter':'1/64'},'exact_parameter_endpoints':['1/64','3/64'],'parameter_center':'1/32','statement':{'period':18,'phase_dimension':13},'settings':{'precision':256,'halfwidth_exponent':-6,'radius_exponent':-21,'newton_steps':2},'passed':True,'gates':{'uniform_root':True,'primitive_period':True,'anchor_identified':True}}
 return prev,c,{'index':0,'passed':True,'gates':copy.deepcopy(c['gates'])}
def test_exact_native_predecessor_join():
 p,c,r=sample();D.validate_row(c,r,p,0)
@pytest.mark.parametrize('mutation',['phase','gap','gauge','rebox','period','index','verdict'])
def test_corrupt_lineage_rejected(mutation):
 p,c,r=sample()
 if mutation=='phase':c['anchor']['state_box']=[2]
 if mutation=='gap':c['exact_parameter_endpoints'][0]='2/64'
 if mutation=='gauge':c['normal']=['-1']
 if mutation=='rebox':c['reboxing']={}
 if mutation=='period':c['statement']['period']=9
 if mutation=='index':r['index']=1
 if mutation=='verdict':r['passed']=False
 with pytest.raises(ValueError):D.validate_row(c,r,p,0)
def test_source_pins_and_unit_prefixes():
 rows=D.O.read(D.ROOT/D.CONFIG)['units'];assert [r['qualified']for r in rows]==[16,16,16,16,12,16,1]
 assert all(r['completed']-r['qualified']in(0,1)for r in rows)
