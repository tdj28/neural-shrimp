import importlib.util
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json,vector_json
from neural_shrimp.rigorous_connection import restore_ball


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_fold_small_march.py'
    s=importlib.util.spec_from_file_location('fold_march',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def chart():
    endpoint=vector_json([arb(0)]*234+[arb(1)])
    return {'passed':True,'gates':{'uniform_root':True,'primitive_period':True,'anchor_identified':True},
      'coordinate_inclusions':[True]*235,'anchor_check':{'passed':True},'statement':{'period':18,'phase_dimension':13},
      'divisors':[{'divisor':d,'excluded':True} for d in (1,2,3,6,9)],
      'settings':{'precision':256,'halfwidth_exponent':-8,'radius_exponent':-21,'newton_steps':2},
      'exact_weights':['1']*235,'normal':['0']*234+['1'],'origin':['0']*235,
      'exact_parameter_endpoints':['1/128','3/128'],'endpoint_root_enclosures':[endpoint,endpoint]}


def fold(rate):return {'orbit':[vector_json([arb(0)]*13) for _ in range(18)],'rate':ball_json(rate)}


def test_projection_directs_count_and_cap():
    m=module();r=m.plan_count(chart(),fold(arb(1)))
    assert r['requested_charts']==16 and r['nominal_remaining_charts']==126
    r=m.plan_count(chart(),fold(arb('1/32')))
    assert r['requested_charts']==2


def test_whole_fold_interval_retained():
    with ctx.workprec(256):
        m=module();r=m.plan_count(chart(),fold(arb('1/32',arb('1/128'))))
        assert restore_ball(r['whole_native0_fold_gauge']).rad()>0
        assert Fraction(r['target_right_gauge'])>Fraction(1,32)+Fraction(1,256)
        assert r['requested_charts']==3


def test_covered_neighborhood_stops_without_claim():
    with pytest.raises(ValueError,match='already covered'):module().plan_count(chart(),fold(arb(0)))


def test_exact_native_shared_endpoint_accepted():
    before=chart();after=chart();after['exact_parameter_endpoints']=['3/128','5/128']
    after['anchor']={'state_box':before['endpoint_root_enclosures'][1],'exact_parameter':'3/128'}
    module().joined(before,after)

@pytest.mark.parametrize('kind',['box','gauge','rate_width','primitive','root'])
def test_invalid_join_or_chart_rejected(kind):
    before=chart();after=chart();after['exact_parameter_endpoints']=['3/128','5/128'];after['anchor']={'state_box':deepcopy(before['endpoint_root_enclosures'][1]),'exact_parameter':'3/128'}
    if kind=='box':after['anchor']['state_box'][0]=ball_json(arb(1))
    if kind=='gauge':after['origin'][0]='1'
    if kind=='rate_width':after['settings']['halfwidth_exponent']=-6
    if kind=='primitive':after['divisors'][0]['excluded']=False
    if kind=='root':after['coordinate_inclusions'][0]=False
    with pytest.raises(ValueError):module().joined(before,after)


def test_parent_pins_required_before_inputs():
    with pytest.raises(ValueError,match='pins'):module().prepare(None,None,None)


def test_repaired_parent_pin_checked_first(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'INITIAL',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='repaired manifest'):m.initial_anchor()
