from fractions import Fraction
import pytest
from flint import arb,arb_mat,arb_series,ctx
from neural_shrimp.rigorous_orbits import exact,ball_json,BallTanhMap
from neural_shrimp.rigorous_folds import FoldTanhMap
from neural_shrimp.rigorous_flips import FlipJet
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_quadratic_connection import path_residual,quadratic_tube,two_quadratic_charts


class QuadraticBranch:
    n=1
    def step(self,x,h):return [x[0]/2+h*h/2],arb_mat([['1/2']]),[h]
    def jet_step(self,x,h):return [x[0]*arb('1/2')+h*h*arb('1/2')]


class TwoCycle:
    n=1
    def step(self,x,h):return [x[0]*x[0]-h],arb_mat([[2*x[0]]]),[-arb(1)]
    def jet_step(self,x,h):return [x[0]*x[0]-h]


def test_exact_quadratic_branch_all_retained_coefficients():
    m=QuadraticBranch();a={'rate':'1','phase_boxes':[[ball_json(arb(1))]]}
    r=quadratic_tube(m,m,[['1']],'1',anchor=a)
    assert r['passed'] and len(r['coordinate_inclusions'])==1
    assert restore_matrix(r['predictor_slope'])[0,0]==2
    assert restore_matrix(r['predictor_curvature'])[0,0]==2
    assert all(restore_matrix(r['center_path_derivatives'][k])[0,0].is_zero() for k in ('f','d','d2','d3'))
    assert restore_matrix(r['whole_path_third_derivative'])[0,0].is_zero()


def test_two_chart_common_parameter_contains_same_exact_root():
    m=QuadraticBranch();inp={'initial':[['1']],'rate':'1','anchor':{'rate':'1','phase_boxes':[[ball_json(arb(1))]]}}
    r=two_quadratic_charts(m,m,inp,halfwidth_exponent=-5,radius_exponent=-24)
    assert r['passed'] and r['charts'][1]['anchor_inclusions']==[True]
    assert r['charts'][0]['exact_parameter_endpoints'][1]==r['charts'][1]['exact_parameter_endpoints'][0]


def test_nonlinear_primitive_two_cycle():
    m=TwoCycle();a={'rate':'1','phase_boxes':[[ball_json(arb(0))],[ball_json(arb(-1))]]}
    r=quadratic_tube(m,m,[['0'],['-1']],'1',anchor=a,halfwidth_exponent=-8,radius_exponent=-20)
    assert r['passed'] and r['divisors'][0]['excluded']
    assert not restore_matrix(r['whole_path_third_derivative'])[0,0].is_zero()


def test_curved_path_third_chain_rule_includes_acceleration():
    class Cube:
        n=1
        def jet_step(self,x,h):return [x[0]*x[0]*x[0]+h*x[0]]
    with ctx.workprec(256):
        # q=2+3d+5d²/2, h=7+d. Residual is q³+hq-q.
        data=path_residual(Cube(),[arb(2)],arb_mat([[3]]),arb_mat([[5]]),arb(7),arb(0),1)
        assert data['f'][0,0]==20
        assert data['d'][0,0]==56
        assert data['d2'][0,0]==204
        assert data['d3'][0,0]==717  # 6q'^3+18q q'q''+3q''


def test_nonzero_proposal_residuals_not_replaced_by_zero():
    m=TwoCycle();r=quadratic_tube(m,m,[['1/100'],['-1']],'1',newton_steps=0,halfwidth_exponent=-8,radius_exponent=-24)
    assert not r['gates']['uniform_root']
    assert not restore_matrix(r['center_path_derivatives']['f'])[0,0].contains(0)


def test_jet_tanh_curved_path_matches_independent_arb_series():
    with ctx.workprec(256):
        a,b,c=map(exact,['1/3','2/5','-1/7'])
        jet=FlipJet(a,[],b,[],c,0).tanh()
        series=arb_series([a,b,c/2],4);ee=(2*series).exp();ref=(ee-1)/(ee+1)
        for i,value in enumerate((jet.f,jet.d,jet.d2,jet.d3)):
            assert value.overlaps(ref[i]*[1,1,2,6][i])


def test_actual_rate_jet_values_match_unchanged_c1_all_coordinates():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','bias':True,'d':1,'width':1,'in_scale':'2','out_scale':'3','loss_scale':'1/2','ridge':'1/100'},'x':[['1/2'],['-1/3']],'y':['1/4','-1/5'],'lr_hidden':'3/8','lr_output':'1/7'}
    with ctx.workprec(256):
        x=list(map(exact,['1/8','-1/4','1/5','1/7']));j=[FlipJet(v,[],arb(0),[],0,0) for v in x]
        rate=FlipJet(exact('3/8'),[],arb(1),[],0,0);value=FoldTanhMap(inp).jet_step(j,rate)
        base,jac=BallTanhMap(inp).step(x)
        assert all(a.f.overlaps(b) for a,b in zip(value,base))
        assert all(a.d.is_zero() for a in value[2:])
        for i in range(2):assert value[i].d.overlaps((base[i]-x[i])/exact('3/8'))


@pytest.mark.parametrize('settings',[{'precision':1},{'newton_steps':3},{'radius_exponent':0},{'halfwidth_exponent':-101}])
def test_invalid_controls(settings):
    m=QuadraticBranch()
    with pytest.raises(ValueError):quadratic_tube(m,m,[['1']],'1',**settings)


def test_anchor_rate_outside_and_wrong_native_root_fail():
    m=QuadraticBranch()
    with pytest.raises(ValueError):quadratic_tube(m,m,[['1']],'1',anchor={'rate':'2','phase_boxes':[[ball_json(arb(4))]]})
    r=quadratic_tube(m,m,[['1']],'1',anchor={'rate':'1','phase_boxes':[[ball_json(arb(2))]]})
    assert not r['passed'] and not r['gates']['anchor_identified']


def test_interruption_stops_calls():
    m=QuadraticBranch();calls=0
    def tick():
        nonlocal calls
        calls+=1
        if calls==3:raise TimeoutError('synthetic')
    with pytest.raises(TimeoutError):quadratic_tube(m,m,[['1']],'1',tick=tick)
    assert calls==3


def driver():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_quadratic_connection.py'
    spec=importlib.util.spec_from_file_location('quadratic_driver_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_profile_counts_and_libraries_fail_closed():
    m=driver();runtime={'python':m.platform.python_version(),'python_flint':m.flint.__version__,'flint':m.flint.__FLINT_VERSION__,
        'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'internal_seconds':20,'tick_cap':300,'map_cap':144,'unit_derivative_cap':288,'jet_cap':108,
        'mode':'profile','threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    result={'elapsed_seconds':1,'max_rss_kib':100,'exit_code':0};m.profile_environment(runtime,result)
    for key in ('map_cap','unit_derivative_cap','jet_cap','tick_cap','numpy'):
        old=runtime[key];runtime[key]='wrong'
        with pytest.raises(ValueError):m.profile_environment(runtime,result)
        runtime[key]=old
    result['elapsed_seconds']=-1
    with pytest.raises(ValueError):m.profile_environment(runtime,result)


def test_profile_and_anchor_identity_fail_before_target(tmp_path,monkeypatch):
    m=driver();(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='profile manifest'):m.profile_gate(tmp_path,'0'*64)
    monkeypatch.setattr(m,'ROOT',tmp_path);p=tmp_path/m.RECEIPT;p.parent.mkdir();p.write_text('{}')
    with pytest.raises(ValueError,match='canonical anchor'):m.prepare()


def test_jet_cap_before_109th_acquisition(monkeypatch):
    m=driver();inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','bias':False,'d':1,'width':1,'in_scale':'1','out_scale':'1','loss_scale':'1/2','ridge':'0'},'x':[['0']],'y':['0'],'lr_hidden':'1','lr_output':'1'}
    model=m.CountedJetMap(inp);model.attempted=108
    monkeypatch.setattr(m.FoldTanhMap,'jet_step',lambda *_:pytest.fail('extra model call'))
    with pytest.raises(ValueError,match='cap'):model.jet_step([],None)


def test_two_chart_full_tick_inventory_under_cap():
    m=QuadraticBranch();calls=0
    def tick():
        nonlocal calls
        calls+=1
    inp={'initial':[['1']],'rate':'1','anchor':{'rate':'1','phase_boxes':[[ball_json(arb(1))]]}}
    result=two_quadratic_charts(m,m,inp,tick=tick)
    # Seven acquisitions per phase per chart, three solves and four observers.
    assert result['passed'] and calls==20
    assert 2*(18*7+3+4)<=300
