from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,exact
from neural_shrimp.rigorous_shifted_arclength import two_shifted_charts
from test_rigorous_arclength_connection import Fold
from test_rigorous_quadratic_connection import TwoCycle


def fold_inputs():
    return {'initial':['5/64','1'],'normal':['1','0'],'origin':['0','1'],'period':1,'weights':['1','1'],
      'anchor':{'state_box':vector_json([exact('5/64'),exact('4121/4096')]),'exact_parameter':'5/64'}}


def test_shifted_chart_preserves_exact_start_and_internal_join():
    r=two_shifted_charts(Fold(),Fold(),fold_inputs(),precision=256,halfwidth_exponent=-8,radius_exponent=-21,newton_steps=2)
    assert r['passed'] and r['connected_parameter_endpoints']==['5/64','3/32']
    a,b=r['charts'];assert a['anchor']['exact_parameter']==a['exact_parameter_endpoints'][0]=='5/64'
    assert a['endpoint_root_enclosures'][1]==b['anchor']['state_box']
    assert a['exact_parameter_endpoints'][1]==b['anchor']['exact_parameter']==b['exact_parameter_endpoints'][0]
    assert a['normal']==b['normal'] and a['origin']==b['origin']


def test_shifted_primitive_two_cycle_native_order():
    inp={'initial':['0','-1','1'],'normal':['0','0','1'],'origin':['0','-1','3/4'],'period':2,'weights':['1']*3,
      'anchor':{'state_box':vector_json([arb(0),arb(-1),arb(1)]),'exact_parameter':'1/4'}}
    r=two_shifted_charts(TwoCycle(),TwoCycle(),inp,precision=256,halfwidth_exponent=-10,radius_exponent=-18,newton_steps=2)
    assert r['passed'] and all(c['divisors'][0]['excluded'] for c in r['charts'])


def test_wrong_inherited_gauge_keeps_failed_first_and_stops():
    inp=fold_inputs();inp['anchor']['exact_parameter']='1'
    r=two_shifted_charts(Fold(),Fold(),inp,precision=256,halfwidth_exponent=-8,radius_exponent=-21,newton_steps=2)
    assert not r['passed'] and len(r['charts'])==1 and not r['charts'][0]['gates']['anchor_identified']


def test_no_loss_of_full_anchor_box():
    inp=fold_inputs();inp['anchor']['state_box'][0]=vector_json([arb('5/64',arb('1/8'))])[0]
    r=two_shifted_charts(Fold(),Fold(),inp,precision=256,halfwidth_exponent=-8,radius_exponent=-21,newton_steps=2)
    assert not r['passed'] and not r['charts'][0]['gates']['anchor_identified']


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_fold_rechart.py'
    s=importlib.util.spec_from_file_location('fold_rechart',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def objects():
    box=vector_json([arb(0,arb('1e-14'))]*234+[arb(75,arb('1e-14'))])
    localized={'passed':True,'scientific_claim':True,'parent_gates':{'root':True},'statement':{'period':18,'phase_dimension':13},'precision':256,'iterations':3,'final_state_box':box}
    previous={'passed':True,'gates':{'root':True},'exact_parameter_endpoints':['31/128','33/128'],'parameter_center':'1/4',
      'normal':['0']*234+['1'],'origin':['0']*234+['74'],'exact_weights':['1/2']*234+['1'],'settings':{'radius_exponent':-15},
      'center':vector_json([arb(0)]*234+[arb(75)]),'predictor_slope':matrix_json(arb_mat([[0]]*234+[[1]])),
      'predictor_curvature':matrix_json(arb_mat(235,1))}
    old={'previous':previous,'model_inputs':{'exact':'same'},'provenance':{'original_anchor':'same'}}
    return localized,old


def test_small_radius_reset_preserves_gauge_weights_and_whole_box():
    loc,old=objects();r=module().inherited_inputs(loc,old)
    assert r['anchor']=={'state_box':loc['final_state_box'],'exact_parameter':'33/128'}
    assert r['normal']==old['previous']['normal'] and r['origin']==old['previous']['origin']
    assert r['weights']==old['previous']['exact_weights']
    assert Fraction(r['initial'][-1])==75+Fraction(3,256)
    assert r['provenance']['radius_reset']=={'old':-15,'new':-21}

@pytest.mark.parametrize('kind',['profile','partial','shape','weight','gauge','parameter'])
def test_bad_parent_scope_rejected(kind):
    loc,old=objects()
    if kind=='profile':loc['scientific_claim']=False
    if kind=='partial':loc['iterations']=2
    if kind=='shape':loc['final_state_box'].pop()
    if kind=='weight':old['previous']['exact_weights'][0]='0'
    if kind=='gauge':old['previous']['normal'][0]='1/3'
    if kind=='parameter':old['previous']['exact_parameter_endpoints'][1]='1'
    with pytest.raises(ValueError):module().inherited_inputs(loc,old)
