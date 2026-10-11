import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import exact,ball_json
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_arclength_connection import bordered,arc_path,arclength_tube,two_arclength_charts,anchor_containment

class Fold:
    n=1
    def step(self,x,h):
        return [x[0]+x[0]*x[0]-(h-1)],arb_mat([[1+2*x[0]]]),[arb(-1)]
    def jet_step(self,x,h):return [x[0]+x[0]*x[0]-(h-1)]


def inputs():
    return {'initial':['0','1'],'normal':['1','0'],'origin':['0','1'],'weights':['1','1'],
        'period':1,'anchor':{'state_box':[ball_json(arb(0)),ball_json(arb(1))]}}


def test_exact_fold_chart_crosses_fixed_parameter_singularity():
    model=Fold();r=two_arclength_charts(model,model,inputs())
    assert r['passed'] and len(r['charts'])==2
    first=r['charts'][0]
    assert first['anchor_check']['kind'].startswith('correlated')
    assert restore_matrix(first['predictor_curvature'])[1,0]==2
    assert restore_matrix(first['predictor_slope'])[0,0]==1
    assert restore_matrix(first['jacobian_tube'])[0,0].contains(0)
    assert all(first['coordinate_inclusions'])
    assert r['charts'][1]['anchor_check']['kind'].startswith('prior certified')


def test_curved_hidden_rate_path_cancels_exact_fold_residual():
    with ctx.workprec(256):
        result=arc_path(Fold(),[arb(0),arb(1)],arb_mat([[1],[0]]),arb_mat([[0],[2]]),
                        [arb(1),arb(0)],[arb(0),arb(1)],arb(0),arb(0),1)
        for v in result.values():assert all(v[i,0]==0 for i in range(2))


def test_nonlinear_parameter_third_derivative():
    class CubicRate(Fold):
        def jet_step(self,x,h):return [x[0]+h*h*h]
    with ctx.workprec(256):
        r=arc_path(CubicRate(),[arb(0),arb(2)],arb_mat([[0],[3]]),arb_mat([[0],[5]]),
                   [arb(1),arb(0)],[arb(0),arb(2)],arb(0),arb(0),1)
        assert r['d3'][0,0]==6*3**3+18*2*3*5


def test_border_parameter_column_and_hyperplane_row():
    r,a,_=bordered(Fold(),[arb(2),arb(5)],1,[arb(3),arb(4)],[arb(1),arb(2)],arb(15))
    assert r==arb_mat([[0],[0]]) and a==arb_mat([[4,-1],[3,4]])


def test_correlated_anchor_uses_full_projection_not_overlap():
    common=dict(center=[arb(0),arb(1)],slope=arb_mat([[1],[0]]),curvature=arb_mat([[0],[2]]),
                normal=[arb(1),arb(0)],origin=[arb(0),arb(1)],scenter=arb(0),delta=arb('1/16'),radii=[arb('1/1024')]*2)
    exact_anchor={'state_box':[ball_json(arb('1/32')),ball_json(arb('1025/1024'))]}
    assert anchor_containment(exact_anchor,**common)['passed']
    wrong={'state_box':[ball_json(arb('1/32')),ball_json(arb('17/16'))]}
    assert not anchor_containment(wrong,**common)['passed']
    outside={'state_box':[ball_json(arb('1/8')),ball_json(arb('65/64'))]}
    assert not anchor_containment(outside,**common)['parameter_inside']


def test_anchor_box_width_retained_even_when_midpoint_is_on_curve():
    common=dict(center=[arb(0),arb(1)],slope=arb_mat([[1],[0]]),curvature=arb_mat([[0],[2]]),
                normal=[arb(1),arb(0)],origin=[arb(0),arb(1)],scenter=arb(0),delta=arb('1/16'),radii=[arb('1/1024')]*2)
    wide={'state_box':[ball_json(arb(0,arb('1/32'))),ball_json(arb(1))]}
    assert not anchor_containment(wide,**common)['passed']


@pytest.mark.parametrize('field,value',[('normal',['0','0']),('weights',['0','1']),('weights',['-1','1']),('weights',['2','1']),('initial',['0','-1']),('normal',['1/3','0'])])
def test_bad_chart_data_fail_closed(field,value):
    inp=inputs();inp[field]=value
    with pytest.raises((ValueError,ZeroDivisionError)):
        two_arclength_charts(Fold(),Fold(),inp)


def test_interruption_is_not_pass():
    def stop():raise TimeoutError('synthetic interrupt')
    with pytest.raises(TimeoutError):two_arclength_charts(Fold(),Fold(),inputs(),tick=stop)


def test_primitive_two_cycle_and_native_phase_separation():
    from test_rigorous_quadratic_connection import TwoCycle
    inp={'initial':['0','-1','1'],'normal':['0','0','1'],'origin':['0','-1','1'],'weights':['1','1','1'],
         'period':2,'anchor':{'state_box':[ball_json(arb(0)),ball_json(arb(-1)),ball_json(arb(1))]}}
    r=two_arclength_charts(TwoCycle(),TwoCycle(),inp,halfwidth_exponent=-10,radius_exponent=-18)
    assert r['passed'] and all(c['divisors'][0]['excluded'] for c in r['charts'])


def test_failed_first_chart_stops_followup():
    inp=inputs();inp['anchor']['state_box'][1]=ball_json(arb(2))
    r=two_arclength_charts(Fold(),Fold(),inp)
    assert not r['passed'] and len(r['charts'])==1 and r['charts'][0]['gates']['uniform_root']


def driver():
    spec=importlib.util.spec_from_file_location('arcdriver',Path(__file__).parents[1]/'tools/certify_s2_period18_arclength_connection.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_parent_manifest_pin_before_parent_science_read(tmp_path,monkeypatch):
    m=driver();monkeypatch.setattr(m,'PARENT',tmp_path);monkeypatch.setattr(m,'original_anchor',lambda:{})
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='parent manifest'):m.prepare()


def test_exact_next_anchor_gauge_and_normal_origin_preserved():
    r=two_arclength_charts(Fold(),Fold(),inputs())
    a,b=r['charts']
    assert a['normal']==b['normal'] and a['origin']==b['origin']
    assert b['anchor']['state_box']==a['endpoint_root_enclosures'][1]
    assert b['anchor']['exact_parameter']==a['exact_parameter_endpoints'][1]
    assert b['anchor']['exact_parameter']==b['exact_parameter_endpoints'][0]


def test_all_arc_sources_bound():
    m=driver()
    for name in ('rigorous_arclength_connection.py','rigorous_folds.py','rigorous_flips.py','rigorous_gradient_enclosure.py','models.py','controls.py'):
        assert 'src/neural_shrimp/'+name in m.SOURCES
    assert m.SETTINGS==dict(precision=256,halfwidth_exponent=-6,radius_exponent=-22,newton_steps=2)


@pytest.mark.parametrize('elapsed',[-1,float('nan'),float('inf'),20])
def test_invalid_profile_resource_fail_closed(elapsed):
    with pytest.raises(ValueError,match='profile time'):driver().profile_environment({},dict(elapsed_seconds=elapsed,max_rss_kib=1))
