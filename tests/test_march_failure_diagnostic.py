import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json

def module():
    p=Path(__file__).parents[1]/'tools/analyze_s2_period18_march_failure.py'
    s=importlib.util.spec_from_file_location('march_failure',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def chart():
    return {'preconditioner':matrix_json(arb_mat([[1,0],[0,1]])),
      'predictor_residual':matrix_json(arb_mat([['1/16'],[0]])),
      'defect':matrix_json(arb_mat([['1/2',0],[0,'1/4']])),
      'coordinate_radii':vector_json([arb('1/16')]*2),'center':vector_json([arb(0)]*2),
      'predictor_slope':matrix_json(arb_mat(2,1)),'predictor_curvature':matrix_json(arb_mat(2,1)),
      'anchor':{'exact_parameter':'0','state_box':vector_json([arb(0)]*2)},'parameter_center':'0',
      'coordinate_inclusions':[False,True],'exact_weights':['1','1']}

def test_forcing_propagation_and_positive_floor():
    with ctx.workprec(384):
        m=module();r=m.diagnose(chart())
        assert r['failed_indices']==[0]
        assert r['rows'][0]['forcing_ratio_upper']['display']=='1.0000000000000000000'
        assert all(m.exact(v)>0 for v in r['proposal']['exact_coordinate_radii'])
        assert r['proposal']['expanded_old_coordinates']==[0]

@pytest.mark.parametrize('value',['-1','0'])
def test_invalid_old_radii(value):
    c=chart();c['coordinate_radii'][0]=ball_json(arb(value))
    with pytest.raises(ValueError,match='positive old radii'):module().diagnose(c)

def test_negative_solution_rejected():
    c=chart();c['defect']=matrix_json(arb_mat([[2,0],[0,0]]))
    with pytest.raises(ValueError,match='nonnegative finite'):module().diagnose(c)

def test_anchor_size_enters_proposal():
    c=chart();c['anchor']['state_box'][1]=ball_json(arb('1/2'))
    # The resulting radius would exceed this unit's declared exponent domain.
    with pytest.raises(ValueError,match='proposal range'):module().diagnose(c)

def align_chart():
    return {'passed':True,'statement':{'period':2,'phase_dimension':1},
      'normal':['0','0','1'],'origin':['1','2','3'],'center':vector_json([arb(1),arb(2),arb(3)]),
      'predictor_slope':matrix_json(arb_mat([[0],[0],[1]])),'predictor_curvature':matrix_json(arb_mat(3,1)),
      'coordinate_radii':vector_json([arb('1/4096')]*3),'parameter_center':'0','settings':{'halfwidth_exponent':-4}}

def test_exact_cyclic_phase_only_identification():
    r=module().fold_alignments(align_chart(),[[ball_json(arb(2))],[ball_json(arb(1))]],ball_json(arb(3)))
    assert [v['contained_in_qualified_chart'] for v in r]==[False,True]

def test_wide_box_not_replaced_by_midpoint():
    r=module().fold_alignments(align_chart(),[[ball_json(arb(1,arb('1/1024')))],[ball_json(arb(2))]],ball_json(arb(3)))
    assert not any(v['contained_in_qualified_chart'] for v in r)

def test_unqualified_chart_cannot_establish_identity():
    c=align_chart();c['passed']=False
    r=module().fold_alignments(c,[[ball_json(arb(1))],[ball_json(arb(2))]],ball_json(arb(3)))
    assert not any(v['contained_in_qualified_chart'] for v in r)

def test_projection_boundary_strict():
    c=align_chart();r=module().fold_alignments(c,[[ball_json(arb(1))],[ball_json(arb(2))]],ball_json(arb('49/16')))
    assert not any(v['strict_parameter_inside'] for v in r)

def test_no_model_or_spectral_acquisition(monkeypatch):
    import numpy as np
    def forbidden(*a,**k):raise AssertionError('forbidden numerical acquisition')
    for name in ('eig','eigh','svd'):monkeypatch.setattr(np.linalg,name,forbidden)
    with ctx.workprec(384):assert module().diagnose(chart())['failed_indices']==[0]
