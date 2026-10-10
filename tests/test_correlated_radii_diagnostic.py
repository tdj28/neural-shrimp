import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import matrix_json,vector_json
from test_march_failure_diagnostic import chart as old_chart


def module():
    p=Path(__file__).parents[1]/'tools/analyze_s2_period18_correlated_radii.py'
    s=importlib.util.spec_from_file_location('correlated_diag',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def chart():
    c=old_chart();c['normal']=['0','1'];c['origin']=['0','0'];c['anchor'].pop('exact_parameter')
    c['coordinate_radii']=vector_json([arb('1/65536')]*2);c['predictor_residual']=matrix_json(arb_mat([['1/262144'],[0]]))
    return c


def test_positive_proposal_and_old_domain_norm():
    with ctx.workprec(256):
        m=module();r=m.diagnose(chart())
        assert all(m.exact(v)>0 for v in r['proposal']['exact_coordinate_radii'])
        assert m.restore_ball(r['old_domain_proposed_weight_norm'])<1


def test_full_correlated_anchor_changes_needed_radius():
    c=chart();c['anchor']['state_box']=vector_json([arb(0),arb(0,arb('1/65536'))])
    c['predictor_slope']=matrix_json(arb_mat([[2],[1]]));r=module().diagnose(c)
    assert module().restore_ball(r['rows'][0]['anchor_required_radius'])>=arb('1/32768')
    assert module().restore_ball(r['full_correlated_projection']).rad()>0


def test_exact_parameter_shortcut_is_rejected():
    c=chart();c['anchor']['exact_parameter']='0'
    with pytest.raises(ValueError,match='correlated'):module().diagnose(c)


def test_singular_proposal_preserves_prior_rows():
    c=chart();c['defect']=matrix_json(arb_mat([[1,0],[0,1]]));rows=[]
    with pytest.raises(Exception):module().diagnose(c,rows.append)
    assert len(rows)==1 and len(rows[0]['rows'])==2


def test_negative_proposal_rejected():
    c=chart();c['defect']=matrix_json(arb_mat([[2,0],[0,0]]))
    with pytest.raises(ValueError,match='nonnegative'):module().diagnose(c)


def test_full_anchor_dimension_required():
    c=chart();c['anchor']['state_box'].pop()
    with pytest.raises(ValueError,match='full correlated'):module().diagnose(c)
