import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_wider_chain.py';s=importlib.util.spec_from_file_location('wider_chain',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
    m=driver();g=dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True)
    c={'schema':'neural-shrimp-quadratic-eigenpair-locus-v1','passed':True,'gates':g,'period':18,'dimension':13,'sigma':1,'shooting_dimension':470,'parameter_center':'37/128','parameter_endpoints':['35/128','39/128'],'normal':['N'],'origin':['O'],'center':[],'slope':[],'curvature':[],'weights':['1'],'settings':{'halfwidth_exponent':-6,'precision':256,'newton_steps':2},'endpoint_root_boxes':[]}
    last={'call':7,'index':3,'right':'39/128'}
    out={'passed':True,'status':'four_joined_charts','calls':8,'repairs':4,'progress':{'accepted':[{}, {}, {},last],'last_accepted':last,'active_call':None}}
    res={'completed':True,'accepted':[1,2,3,4],'calls':8,'repairs':4,'attempts':[{'call':i,'passed':bool(i%2)} for i in range(8)],'last_chart':copy.deepcopy(c)}
    a={'call':7,'index':3,'chart':copy.deepcopy(c)}
    return m,out,res,a,c,copy.deepcopy(c)


def test_complete_parent_binding():
    m,o,r,a,c,anchor=fixture();assert m.validate_parent(o,r,a,c,'39/128',anchor)==c


@pytest.mark.parametrize('case',['partial','failed','wrong_gauge','wrong_width','wrong_right','extra_call','unpersisted','mismatched_full'])
def test_reject_unusable_parent(case):
    m,o,r,a,c,anchor=fixture()
    if case=='partial':r['completed']=False
    if case=='failed':c['passed']=False
    if case=='wrong_gauge':anchor['normal']=['different']
    if case=='wrong_width':c['settings']['halfwidth_exponent']=-7
    if case=='wrong_right':c['parameter_endpoints'][1]='40/128'
    if case=='extra_call':r['calls']=9
    if case=='unpersisted':o['progress']['active_call']=7
    if case=='mismatched_full':c['origin']=['wrong']
    with pytest.raises(ValueError):m.validate_parent(o,r,a,c,'39/128',anchor)


def test_cli_and_transitive_inventory():
    m=driver();a=SimpleNamespace(unit=2,expected_right='39/128',parent=m.T.BASE/'attempt-001-wider-march',parent_source='a'*40,parent_manifest='b'*64,parent_run_manifest='c'*64)
    m.install(a)
    for k,v in [('unit',1),('expected_right','40/128'),('parent',Path('/tmp/wrong')),('parent_source','x')]:
        b=copy.copy(a);setattr(b,k,v)
        with pytest.raises(ValueError):m.install(b)
    root=Path(__file__).parents[1];assert set(m.W.SOURCES)<=set(m.SOURCES) and all((root/n).is_file() for n in m.SOURCES)
