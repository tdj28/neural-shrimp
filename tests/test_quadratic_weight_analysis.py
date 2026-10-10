import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import matrix_json,ball_json,exact
from neural_shrimp.rigorous_connection import restore_ball


def module():
    path=Path(__file__).resolve().parents[1]/'tools/analyze_s2_period18_quadratic_weights.py'
    spec=importlib.util.spec_from_file_location('weight_analysis_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def chart():
    return {'passed':True,'gates':{'uniform_root':True},'preconditioner':matrix_json(arb_mat([[1,0],[0,1]])),
        'predictor_residual':matrix_json(arb_mat([[0],[0]])),
        'defect':matrix_json(arb_mat([['1/5','3/10'],['1/100','1/50']])),
        'correction_radius':ball_json(arb(1))}


def test_exact_weighted_norm_and_separated_forcing():
    with ctx.workprec(384):
        m=module();q=m.reduce_chart(chart(),['1','1/4'])
        upper=restore_ball(q['weighted_norm_upper'])
        assert upper>=exact('11/40') and upper<exact('276/1000')
        assert q['contraction_passed'] and q['inclusion_passed']
        assert restore_ball(q['weighted_forcing_ratio_upper']).is_zero()
        assert restore_ball(q['weighted_propagated_ratio_upper'])<exact('276/1000')


def test_forcing_failure_is_distinct_from_contraction():
    m=module();c=chart();c['predictor_residual']=matrix_json(arb_mat([[2],[0]]))
    q=m.reduce_chart(c,['1','1/4']);assert q['contraction_passed'] and not q['inclusion_passed']


def test_one_positive_float_proposal_checked_by_balls():
    m=module();q=m.reduce_chart(chart());assert q['contraction_passed'] and q['inclusion_passed']
    assert len(q['exact_weights'])==2 and q['subdomain_of_parent']


@pytest.mark.parametrize('weights',[['0','1'],['-1','1'],['2','1'],['1'],['1/3','1']])
def test_invalid_nonpoint_or_nonpositive_weights(weights):
    # 1/3 is exact-real rational but not a point dyadic at finite Arb precision.
    with pytest.raises(ValueError):module().reduce_chart(chart(),weights)


def test_parent_failure_rejected():
    c=chart();c['passed']=False
    with pytest.raises(ValueError):module().reduce_chart(c)
