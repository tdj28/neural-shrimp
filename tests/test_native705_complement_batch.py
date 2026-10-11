import copy,json,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_complement_batch as B


def fixture(unit=1):
    b=B.D.ball_json;c=[arb(0)]*702+[arb(1),arb(1),arb('1/16')];rho=arb((1,-10));zero=b(arb(0));sc,ends=B.geometry(unit)
    return dict(schema='neural-shrimp-native-unit-pair-pal-v1',dimension=13,period=18,shooting_dimension=705,settings=dict(precision=256,newton_steps=2,halfwidth_exponent=-7,radius_exponent=-10),parameter_center=sc,parameter_endpoints=ends,gates=dict(uniform_full_native_unit_pair_locus=True,primitive_period=True,realified_frame_independent=True,anchor_identified=False),center=[b(v)for v in c],slope=[zero]*705,curvature=[zero]*705,weights=['1']*705,coordinate_radii=[b(rho)]*705,whole_box=[b(v+arb(0,rho))for v in c],preconditioner=[[b(arb(1))]],whole_jacobian=[[b(arb(1))]],center_residual=[[zero]]*705,quadratic_center_coefficients=[[zero]*4]*705,quadratic_whole_coefficients=[[zero]*4]*705,predictor_residual=[[zero]]*705)


def test_four_exact_gapless_preproved_intervals_only():
    rows=[B.geometry(i)for i in range(1,5)];assert rows[0][1][0]=='63/4096'and rows[-1][1][1]=='319/4096'
    assert all(rows[i][1][1]==rows[i+1][1][0]for i in range(3))
    for bad in [0,5,True,1.0]:
        with pytest.raises(ValueError):B.geometry(bad)


def test_exact_chart_metadata_and_full_weight_binding():
    with ctx.workprec(384):
        chart=fixture();args=B.chart_inputs(chart,1);assert len(args['center'])==705
        with pytest.raises(ValueError):B.chart_inputs(chart,2)
        bad=copy.deepcopy(chart);bad['coordinate_radii'][703]=B.D.ball_json(arb((1,-9)))
        with pytest.raises(ValueError,match='radius'):B.chart_inputs(bad,1)
        bad=copy.deepcopy(chart);bad['gates']['primitive_period']=False
        with pytest.raises(ValueError):B.chart_inputs(bad,1)


def test_exact_original_domain_never_widened():
    with ctx.workprec(384):
        chart=fixture();chart['whole_box'][702]=B.D.ball_json(arb(1,arb((1,-11))))
        with pytest.raises(ValueError,match='original derivative domain'):B.chart_inputs(chart,1)


def test_original_residual_not_replaced_by_uncovered_decode():
    with ctx.workprec(384):
        chart=fixture();chart['quadratic_center_coefficients'][0]=[B.D.ball_json(arb((1,-100)))]+[B.D.ball_json(arb(0))]*3
        with pytest.raises(ValueError,match='predictor residual'):B.chart_inputs(chart,1)


def test_entire_angle_domain_required():
    with ctx.workprec(384):
        chart=fixture();chart['center'][704]=B.D.ball_json(arb(0));chart['whole_box'][704]=B.D.ball_json(arb(0,arb((1,-10))))
        with pytest.raises(ValueError,match='positive'):B.chart_inputs(chart,1)


def test_failure_can_consume_closing_reserve(tmp_path):
    used=[124*1024**2-10]
    with pytest.raises(ValueError):B.save_bound(tmp_path,'too-large.json',dict(v='x'*100),used)
    B.save_bound(tmp_path,'outcome.json',dict(passed=False),used,True);B.save_bound(tmp_path,'manifest.json',dict(files={'outcome.json':B.O.entry(tmp_path/'outcome.json')}),used,True)


def test_isolated_committed_source_closure(tmp_path):
    for n in B.FILES:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    code='import sys;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import probe_native705_complement_batch as b;assert b.ROOT==__import__("pathlib").Path('+repr(str(tmp_path))+');assert all((b.ROOT/n).is_file() for n in b.FILES)'
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
