import copy,importlib.util
from pathlib import Path
import pytest


def driver():
    spec=importlib.util.spec_from_file_location('marchdriver',Path(__file__).parents[1]/'tools/certify_s2_period18_arclength_march.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fake_chain():
    m=driver()
    base={'passed':True,'gates':{'uniform_root':True,'primitive_period':True,'anchor_identified':True},
          'settings':m.SETTINGS,'statement':{'period':18,'phase_dimension':13},'normal':['fixed'],'origin':['fixed'],
          'exact_weights':['fixed'],'exact_parameter_endpoints':['0','1'],'endpoint_root_enclosures':[['left'],['right0']],
          'center':[],'parameter_center':'1/2','predictor_slope':{},'predictor_curvature':{}}
    charts=[];previous=copy.deepcopy(base)
    for i in range(16):
        c=copy.deepcopy(previous);c.update(anchor={'state_box':previous['endpoint_root_enclosures'][1],'exact_parameter':str(i+1)},
            anchor_check={'parameter_inside':True,'coordinate_inclusions':[True]*235},coordinate_inclusions=[True]*235,
            exact_parameter_endpoints=[str(i+1),str(i+2)],endpoint_root_enclosures=[['left'+str(i)],['right'+str(i+1)]],
            divisors=[{'divisor':d,'excluded':True} for d in (1,2,3,6,9)])
        charts.append(c);previous=c
    rows=[{'index':i,'passed':True} for i in range(16)]
    return base,rows,charts


def test_complete_saved_chain_join_validation_without_acquisition():
    base,rows,charts=fake_chain();m=driver();last=m.validate_saved_chain(base,rows,lambda row:charts[row['index']])
    assert last['endpoint_root_enclosures']==charts[-1]['endpoint_root_enclosures']


@pytest.mark.parametrize('mutation',['normal','origin','weights','anchor_box','anchor_parameter','settings','primitive','root','inclusion','order','short'])
def test_saved_parent_chain_contradictions_rejected(mutation):
    base,rows,charts=fake_chain();m=driver();c=charts[7]
    if mutation=='normal':c['normal']=['other']
    if mutation=='origin':c['origin']=['other']
    if mutation=='weights':c['exact_weights']=['other']
    if mutation=='anchor_box':c['anchor']['state_box']=['other']
    if mutation=='anchor_parameter':c['anchor']['exact_parameter']='-1'
    if mutation=='settings':c['settings']['radius_exponent']=-21
    if mutation=='primitive':c['divisors'][0]['excluded']=False
    if mutation=='root':c['gates']['uniform_root']=False
    if mutation=='inclusion':c['anchor_check']['coordinate_inclusions'][4]=False
    if mutation=='order':rows[7]['index']=8
    if mutation=='short':rows.pop()
    with pytest.raises(ValueError):m.validate_saved_chain(base,rows,lambda row:charts[row['index']])


def test_pins_required_before_any_reference_reads(tmp_path):
    m=driver()
    for pin,commit in [('', 'a'*40),('a'*64,''),('nothex'*11,'b'*40)]:
        with pytest.raises(ValueError,match='predecessor pins'):m.prepare(tmp_path,pin,commit)


def test_both_accepted_producer_drivers_and_all_math_bound():
    m=driver()
    assert 'tools/certify_s2_period18_arclength_batch.py' in m.SOURCES
    assert 'tools/certify_s2_period18_arclength_march.py' in m.SOURCES
    assert set(m.PROOF_SOURCES)<=set(m.SOURCES)
