from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from flint import arb
from neural_shrimp.rigorous_orbits import vector_json
from test_s2_period18_fold_march import chart


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_fold_saved_rebox.py'
    s=importlib.util.spec_from_file_location('saved_rebox',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
    a=chart();a['settings']['halfwidth_exponent']=-8;a['exact_parameter_endpoints']=['33/128','17/64'];a['anchor']={'state_box':vector_json([arb(0)]*235),'exact_parameter':'33/128'}
    a['known_fold_membership']={'all_cyclic_alignments':[{'contained_in_qualified_chart':False} for _ in range(18)]}
    b=deepcopy(a);b['passed']=False;b['gates']['uniform_root']=False;b['exact_parameter_endpoints']=['17/64','35/128'];b['anchor']={'state_box':deepcopy(a['endpoint_root_enclosures'][1]),'exact_parameter':'17/64'}
    inp={'anchor':a['anchor'],'normal':a['normal'],'origin':a['origin'],'weights':a['exact_weights']}
    return {'passed':False,'charts':[a,b]},inp


def test_only_failed_second_witness_selected():
    result,inp=fixture();assert module().target_pair(result,inp) is result['charts'][1]

@pytest.mark.parametrize('kind',['first_unproved','second_pass','second_primitive','phase','parameter','normal','settings','divisor','fold'])
def test_saved_pair_tamper_rejected(kind):
    r,i=fixture();a,b=r['charts']
    if kind=='first_unproved':a['coordinate_inclusions'][0]=False
    if kind=='second_pass':b['passed']=True
    if kind=='second_primitive':b['gates']['primitive_period']=False
    if kind=='phase':b['anchor']['state_box'][0]['mid']=[1,0]
    if kind=='parameter':b['anchor']['exact_parameter']='0'
    if kind=='normal':b['normal'][0]='1'
    if kind=='settings':b['settings']['precision']=128
    if kind=='divisor':b['divisors'][0]['excluded']=False
    if kind=='fold':b['known_fold_membership']['all_cyclic_alignments'][0]['contained_in_qualified_chart']=True
    with pytest.raises(ValueError):module().target_pair(r,i)


def test_engineering_proposal_is_exact_and_expands_box():
    m=module();p=m.synthetic_proposal({'exact_weights':['1','1/2']})
    assert p['radius_exponent']==-20 and p['exact_coordinate_radii']==['1/1048576','1/2097152']


def test_driver_durable_save_roundtrip(tmp_path):
    m=module();payload=m.save(tmp_path/'test.json.gz',{'a':'saved'})
    m.save(tmp_path/'test-payload.json',payload)
    assert m.read(tmp_path/'test.json.gz')=={'a':'saved'}
