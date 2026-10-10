import importlib.util
from pathlib import Path
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import exact,ball_json
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_weighted_connection import weighted_quadratic_tube,two_weighted_quadratic_charts
from test_rigorous_quadratic_connection import QuadraticBranch,TwoCycle


def test_weighted_exact_branch_and_shared_endpoint():
    model=QuadraticBranch();inp={'initial':[['1']],'rate':'1','anchor':{'rate':'1','phase_boxes':[[ball_json(arb(1))]]}}
    r=two_weighted_quadratic_charts(model,model,inp,weights=['1/8'])
    assert r['passed']
    for chart in r['charts']:
        assert chart['exact_weights']==['1/8']
        assert restore_ball(chart['coordinate_radii'][0])==restore_ball(chart['correction_radius'])/8


def test_weighted_anchor_uses_coordinate_radius():
    model=QuadraticBranch()
    r=weighted_quadratic_tube(model,model,[['1']],'1',weights=['1/8'],radius_exponent=-8,
        anchor={'rate':'1','phase_boxes':[[ball_json(exact('1025/1024'))]]})
    assert r['gates']['uniform_root'] and not r['gates']['anchor_identified']


def test_nonlinear_weighted_norm_and_true_domain():
    with ctx.workprec(256):
        model=TwoCycle();r=weighted_quadratic_tube(model,model,[['0'],['-1']],'1',weights=['1','1/2'],halfwidth_exponent=-10,radius_exponent=-18,
            anchor={'rate':'1','phase_boxes':[[ball_json(arb(0))],[ball_json(arb(-1))]]})
        assert r['passed'];d=restore_matrix(r['defect']);w=[arb(1),arb('1/2')]
        expected=max((sum(abs(d[i,j]).upper()*w[j] for j in range(2))/w[i]).upper() for i in range(2))
        # The public ball constructor rounds a restored radius outward. Compare
        # the independently recomputed formula to that one-radius-ulp scale;
        # this regression check is not a certificate replay.
        assert abs(expected-restore_ball(r['contraction'])).upper()<exact('1/10000000000')
        assert len(r['phase_jacobians'])==2 and all(r['coordinate_inclusions'])


@pytest.mark.parametrize('w',[['0'],['-1'],['2'],['1/3'],[],['1','1'],[.5]])
def test_strict_dyadic_positive_weights(w):
    m=QuadraticBranch()
    with pytest.raises(ValueError):weighted_quadratic_tube(m,m,[['1']],'1',weights=w)


def driver():
    path=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_weighted_connection.py'
    spec=importlib.util.spec_from_file_location('weighted_driver_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_weight_manifest_pin_before_read(tmp_path,monkeypatch):
    m=driver();monkeypatch.setattr(m,'WEIGHTS',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='weight manifest'):m.prepare_weights()


def test_all_weight_sources_bound():
    m=driver()
    assert 'src/neural_shrimp/rigorous_weighted_connection.py' in m.SOURCES
    assert 'src/neural_shrimp/rigorous_quadratic_connection.py' in m.SOURCES
    assert 'src/neural_shrimp/rigorous_folds.py' in m.SOURCES
    assert 'src/neural_shrimp/rigorous_flips.py' in m.SOURCES
    assert m.SETTINGS['halfwidth_exponent']==-6 and m.SETTINGS['radius_exponent']==-24
