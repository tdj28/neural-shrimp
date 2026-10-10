"""Synthetic strict parent-descriptor joins; no scientific artifact reads."""
import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def load():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_locus_chain.py';s=importlib.util.spec_from_file_location('chain_adapter',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
    z={'mid':[0,0],'rad':[1,-40]};size=470
    chart={'schema':'neural-shrimp-quadratic-eigenpair-locus-v1','passed':True,'gates':dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True),'period':18,'dimension':13,'sigma':1,'shooting_dimension':size,'parameter_center':'5/16384','parameter_endpoints':['9/32768','11/32768'],'normal':['0']*469+['1'],'origin':['0']*size,'center':[z]*size,'slope':[z]*size,'curvature':[z]*size,'weights':['1']*size,'settings':{'precision':256,'newton_steps':2,'halfwidth_exponent':-15,'radius_exponent':-18},'endpoint_root_boxes':[[z]*size,[z]*size]}
    records=[{'call':i,'index':i,'right':str((5+2*i))+'/32768','full_chart_file':f'events/{i}-chart.json.gz','accepted_file':f'events/{i}-accepted.json.gz'} for i in range(4)]
    outcome={'passed':True,'status':'four_joined_charts','calls':4,'progress':{'accepted':records,'last_accepted':records[-1],'active_call':None}}
    result={'completed':True,'calls':4,'accepted':[{}]*4,'last_chart':copy.deepcopy(chart)}
    accepted={'call':3,'index':3,'chart':copy.deepcopy(chart)}
    return outcome,result,accepted,chart


def test_full_chart_compact_and_durable_record_agree():
    m=load();o,r,a,c=fixture();assert m.validate_completion(o,r,a,c,'11/32768')==c
    assert m.input_path(Path('/tmp/parent'),'events/1-chart.json.gz')==Path('/tmp/parent/run/events/1-chart.json.gz')
    for value in ('/tmp/x','../x','a/../../x'):
        with pytest.raises(ValueError):m.input_path(Path('/tmp/parent'),value)


@pytest.mark.parametrize('which,key,value',[
 ('out','passed',False),('out','status','partial'),('result','completed',False),('result','calls',6),
 ('accepted','call',2),('accepted','index',2),('chart','passed',False),('chart','sigma',-1),('chart','period',17),('chart','dimension',12),('chart','shooting_dimension',469)])
def test_bad_or_unjoined_parent_rejects(which,key,value):
    m=load();o,r,a,c=fixture();target={'out':o,'result':r,'accepted':a,'chart':c}[which];target[key]=value
    with pytest.raises(ValueError):m.validate_completion(o,r,a,c,'11/32768')


def test_chart_mismatch_pending_call_wrong_fixed_grid_and_missing_record_reject():
    m=load()
    for mutate in (lambda o,r,a,c:o['progress'].update(active_call=4),lambda o,r,a,c:r['last_chart']['weights'].__setitem__(0,'1/2'),lambda o,r,a,c:a['chart']['origin'].__setitem__(0,'1'),lambda o,r,a,c:o['progress']['accepted'].pop()):
        o,r,a,c=fixture();mutate(o,r,a,c)
        with pytest.raises(ValueError):m.validate_completion(o,r,a,c,'11/32768')
    o,r,a,c=fixture()
    with pytest.raises(ValueError):m.validate_completion(o,r,a,c,'19/32768')


def test_install_keeps_old_sources_and_sets_explicit_parent_only(monkeypatch):
    m=load();old=m.ORIGINAL_SOURCES.copy();args=SimpleNamespace(mode='target',unit=2,parent=m.BASE/'attempt-001-march-target',parent_source='a'*40,parent_manifest='b'*64,parent_run_manifest='c'*64,expected_right='11/32768')
    original=m.M.bind_parent
    try:
        m.install(args);assert set(old+m.EXTRA)==set(m.M.SOURCES) and m.M.bind_parent is not original
    finally:m.M.bind_parent=original;m.M.SOURCES=old
    for field,value in [('unit',1),('parent',m.BASE/'attempt-002-march-target'),('parent_source','bad'),('parent_manifest','bad'),('expected_right','19/32768')]:
        bad=copy.copy(args);setattr(bad,field,value)
        with pytest.raises(ValueError):m.install(bad)
    m.M.bind_parent=original;m.M.SOURCES=old


def test_sources_exist_and_launcher_requires_all_prospective_pins():
    m=load();root=Path(__file__).parents[1];assert all((root/n).is_file() for n in m.ORIGINAL_SOURCES+m.EXTRA)
    s=(root/'tools/launch_eigenpair_chain.py').read_text()
    for key in ('parent-source','parent-manifest','parent-run-manifest','expected-right','profile-outer-manifest'):assert key in s
    assert '315 if target else 25' in s and 'attempt-001-chain-profile' in s
