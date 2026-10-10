import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat
from neural_shrimp.rigorous_orbits import matrix_json,ball_json


def module():
    p=Path(__file__).parents[1]/'tools/analyze_s2_period18_fold_reverse_restriction.py';s=importlib.util.spec_from_file_location('restriction_diag',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def chart():
    return {'gates':{'uniform_root':False,'primitive_period':True,'anchor_identified':True},'settings':{'halfwidth_exponent':-7},'parameter_center':'0',
      'preconditioner':matrix_json(arb_mat([[1]])),'defect':matrix_json(arb_mat([[2]])),'exact_weights':['1'],
      'coordinate_radii':[ball_json(arb((1,-16)))],'predictor_residual':matrix_json(arb_mat([[arb((1,-15))]])),
      'center_path_derivatives':{k:matrix_json(arb_mat([[arb((1,-30)) if k=='f' else 0]])) for k in ('f','d','d2','d3')},
      'whole_path_third_derivative':matrix_json(arb_mat([[6]]))}


def test_all_defects_retained_and_radius_exact():
    m=module();r=m.diagnose(chart())
    assert m.restore_ball(r['weighted_taylor_part_upper'][0])>=arb((1,-30))
    assert m.restore_ball(r['weighted_taylor_part_upper'][3])>=arb((1,-30))
    assert arb((1,r['radius_exponent']))>=8*m.restore_ball(r['weighted_forcing_upper'])
    assert r['new_halfwidth_exponent']==-10 and 'fresh' in r['scope']

@pytest.mark.parametrize('key',['uniform_root','primitive_period','anchor_identified'])
def test_wrong_gate_rejected(key):
    c=chart();c['gates'][key]=not c['gates'][key]
    with pytest.raises(ValueError,match='root-only'):module().diagnose(c)
