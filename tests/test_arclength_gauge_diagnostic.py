import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json


def module():
    path=Path(__file__).parents[1]/'tools/analyze_s2_period18_gauge.py'
    spec=importlib.util.spec_from_file_location('gauge_diag',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def chart():
    return {'passed':True,'gates':{'root':True},'endpoint_root_enclosures':[[],vector_json([arb(2,arb('1/1024')),arb(3)])],
      'predictor_slope':matrix_json(arb_mat([[3],[4]])),'predictor_curvature':matrix_json(arb_mat(2,1)),
      'settings':{'halfwidth_exponent':-6},'normal':['0','1'],'jacobian_tube':matrix_json(arb_mat([[1,0],[0,1]]))}


def test_fixed_dyadic_forward_normal_and_full_endpoint():
    with ctx.workprec(256):
        m=module();r=m.gauge_proposal(chart());n=[m.exact(v) for v in r['normal']]
        assert all(v.is_exact() for v in n)
        assert n[0]>0 and n[1]>0
        assert m.restore_ball(r['endpoint_gauge_box']).rad()>0
        assert m.restore_ball(r['max_linearized_anchor_radius'])>0
        assert r['origin']==['2','3']

@pytest.mark.parametrize('field',['passed','gates'])
def test_unqualified_parent_rejected(field):
    c=chart();c[field]=False if field=='passed' else {'root':False}
    with pytest.raises(ValueError,match='qualified'):module().gauge_proposal(c)


def test_zero_tangent_rejected():
    c=chart();c['predictor_slope']=matrix_json(arb_mat(2,1))
    with pytest.raises(ValueError,match='tangent length'):module().gauge_proposal(c)


def test_curvature_contributes_at_endpoint():
    c=chart();c['predictor_curvature']=matrix_json(arb_mat([[64],[0]]));r=module().gauge_proposal(c)
    assert module().restore_ball(r['endpoint_taylor_tangent'][0])==4


def test_midpoint_border_replaced_only_last_row():
    m=module();r=m.bordered_diagnostics(chart(),[arb(1),arb(1)])
    assert r['diagnostics'][0]['invertible_saved_midpoint_matrix']
    assert r['diagnostics'][1]['invertible_saved_midpoint_matrix']
    assert m.restore_ball(r['diagnostics'][1]['condition_inf_upper'])>=4


def test_singular_new_border_retained():
    r=module().bordered_diagnostics(chart(),[arb(1),arb(0)])
    assert not r['diagnostics'][1]['invertible_saved_midpoint_matrix']


def test_all_cyclic_alignments_and_full_radius_retained():
    m=module();r=m.fold_projections([[ball_json(arb(1,arb('1/8')))],[ball_json(arb(2))]],ball_json(arb(3)),
      [arb(1),arb(0),arb(0)],[arb(0)]*3,arb(0,arb('1/16')),arb('1/64'))
    assert [v['cyclic_shift'] for v in r['alignments']]==[0,1]
    assert m.restore_ball(r['alignments'][0]['projected_gap_box']).rad()>arb('1/8')
    assert 'no connection' in r['scope']


def test_project_requires_all_coordinates():
    with pytest.raises(ValueError,match='dimensions'):module().project([arb(0)],[arb(1),arb(1)],[arb(0)])
