import importlib.util
from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_chart_restriction import restrict_chart
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_connection_frames import restore_matrix
from test_rigorous_arclength_connection import Fold,inputs


def fixture():
    c=two_arclength_charts(Fold(),Fold(),inputs(),halfwidth_exponent=-7,radius_exponent=-16)['charts'][0]
    p={'halfwidth_exponent':-10,'radius_exponent':-18,'exact_weights':['1','1'],'exact_coordinate_radii':[str(Fraction(1,2**18))]*2}
    return c,p


def test_correlated_fold_root_retained_with_new_domain_and_broad_h3():
    with ctx.workprec(256):
        c,p=fixture();r=restrict_chart(Fold(),c,p)
        assert r['passed'] and 'exact_parameter' not in r['anchor']
        assert r['exact_parameter_endpoints']==['-1/1024','1/1024']
        assert all(r['restriction']['path_containment'])
        assert r['whole_path_third_derivative']==c['whole_path_third_derivative']
        assert r['center_path_derivatives']==c['center_path_derivatives']
        assert restore_matrix(r['jacobian_tube'])[0,0].rad()<restore_matrix(c['jacobian_tube'])[0,0].rad()

@pytest.mark.parametrize('change',['width','weights','radii','anchor'])
def test_restriction_safety(change):
    c,p=fixture()
    if change=='width':p['halfwidth_exponent']=-6
    if change=='weights':p['exact_weights'][0]='0'
    if change=='radii':p['exact_coordinate_radii'][0]='1/2'
    if change=='anchor':c['anchor']['state_box'].pop()
    with pytest.raises(ValueError):restrict_chart(Fold(),c,p)


def test_lower_taylor_defect_never_zeroed():
    c,p=fixture();c['center_path_derivatives']['f'][0][0]={'mid':[1,-10],'rad':[0,0],'display':'1/1024'}
    r=restrict_chart(Fold(),c,p)
    assert not r['gates']['uniform_root']
    assert restore_matrix(r['predictor_residual'])[0,0].contains(arb((1,-10)))


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_fold_restricted_chart.py';s=importlib.util.spec_from_file_location('restricted_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_fixed_scope_and_manifest_first(tmp_path,monkeypatch):
    m=module();assert 'src/neural_shrimp/rigorous_chart_restriction.py' in m.SOURCES
    monkeypatch.setattr(m,'PARENT',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):m.prepare()
