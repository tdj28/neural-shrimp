import importlib.util
from pathlib import Path
import pytest
from flint import arb
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json
p=Path(__file__).parents[1]/'tools/probe_eigenpair_chain_unfolding.py';s=importlib.util.spec_from_file_location('coeff_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture():
    box,j=m.synthetic();out=dict(passed=True,status='completed',native_steps=72,positive_actual_rates=True,root_passed=True,primitive_period=18,strict_inclusions=41,quotient_unit_circle_excluded=True,complement_attracting=True,exact_algebraic_multiplicity_two=True)
    fresh={'root':{'passed':True,'coordinate_inclusions':[True]*41,'contraction':ball_json(arb(1)/2)},'divisors':[{'divisor':d,'passed':True}for d in [1,2,3,6,9]],'box':vector_json(box),'whole_jacobian':matrix_json(j)}
    return out,fresh

def test_synthetic_full_size_coefficient_and_admission():
    z,j=m.admit_audit(*fixture());assert len(z)==j.nrows()==j.ncols()==41
    r=m.chain_coefficients(z,j);assert r['quadratic_nondegenerate']and r['resonant_nondegenerate']and r['unfolding_transverse']

@pytest.mark.parametrize('key',['passed','positive_actual_rates','root_passed','quotient_unit_circle_excluded','complement_attracting','exact_algebraic_multiplicity_two'])
def test_all_independent_root_spectrum_prerequisites_required(key):
    out,fresh=fixture();out[key]=False
    with pytest.raises(ValueError):m.admit_audit(out,fresh)

@pytest.mark.parametrize('key',['root','divisor','rate','dimension'])
def test_fresh_box_corruptions_fail(key):
    out,f=fixture()
    if key=='root':f['root']['coordinate_inclusions'][0]=False
    elif key=='divisor':f['divisors'][2]['passed']=False
    elif key=='rate':f['box'][13]=ball_json(arb(-1))
    else:f['whole_jacobian'].pop()
    with pytest.raises(ValueError):m.admit_audit(out,f)

def test_measured_saved_only_profile():
    good={'passed':True,'status':'synthetic_coefficients_completed','ticks':0,'evaluation_seconds':[.01,.02]}
    assert m.cost(good)==15.08
    for bad in [dict(good,ticks=1),dict(good,evaluation_seconds=[float('nan'),.2]),dict(good,passed=False),dict(good,evaluation_seconds=[4,.2])]:
        with pytest.raises(ValueError):m.cost(bad)

def test_actual_launcher_worker_and_fixed_precursor():
    p=Path(__file__).parents[1]/'tools/launch_eigenpair_chain_unfolding.py';s=importlib.util.spec_from_file_location('coeff_launcher',p);l=importlib.util.module_from_spec(s);s.loader.exec_module(l)
    l.ARGS=m.parser().parse_args(['--mode','target','--out',str(m.BASE/'attempt001-target/run'),'--profile','/profile']);l.OUT=m.BASE/'attempt001-target';l.LIMIT=40
    assert str(m.ROOT/'tools/probe_eigenpair_chain_unfolding.py')in l.command()
