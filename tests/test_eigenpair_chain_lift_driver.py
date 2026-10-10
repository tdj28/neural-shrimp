import importlib.util,copy
from pathlib import Path
import pytest
from flint import arb
from neural_shrimp.rigorous_orbits import vector_json,ball_json
p=Path(__file__).parents[1]/'tools/probe_eigenpair_chain_lift.py';s=importlib.util.spec_from_file_location('lift_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_only_complete_measured_profile_admits_target_cost():
    good={'passed':True,'status':'synthetic_lift_completed','ticks':36,'evaluation_seconds':[.1,.2]}
    assert m.cost(good)==15.8
    for bad in [dict(good,ticks=18),dict(good,passed=False),dict(good,status='failed'),dict(good,evaluation_seconds=[.1,float('nan')]),dict(good,evaluation_seconds=[.1,4])]:
        with pytest.raises(ValueError):m.cost(bad)

def fixture():
    gates=dict.fromkeys(('normalized_augmented_root_unique','positive_actual_rates','primitive_period'),True)
    out={'passed':True,'status':'chain_proof_completed','ticks':360,'whole_box_proximity':True,'gates':gates}
    proof={'schema':'neural-shrimp-generalized-chain-root-v1','passed':True,'gates':gates,'period':18,'dimension':13,'augmented_dimension':41,'settings':{'radius_exponent':-80,'newton_steps':8},'box':vector_json([arb(1)]*41),'coordinate_inclusions':[True]*41,'contraction':ball_json(arb(1)/2),'divisors':[{'divisor':d,'excluded':True}for d in [1,2,3,6,9]]}
    return proof,out

def test_exact_root_gate_admission():
    assert len(m.qualified_root(*fixture()))==41

@pytest.mark.parametrize('kind',['period','count','inclusion','contraction','divisor','negative_rate','outcome','proximity'])
def test_target_corruptions_fail_closed(kind):
    proof,out=fixture()
    if kind=='period':proof['period']=9
    elif kind=='count':out['ticks']=359
    elif kind=='inclusion':proof['coordinate_inclusions'][3]=False
    elif kind=='contraction':proof['contraction']=ball_json(arb(1))
    elif kind=='divisor':proof['divisors'][2]['excluded']=False
    elif kind=='negative_rate':proof['box'][14]=ball_json(arb(-1))
    elif kind=='outcome':out['passed']=False
    else:out['whole_box_proximity']=False
    with pytest.raises(ValueError):m.qualified_root(proof,out)

def test_owned_launcher_has_actual_lift_worker_and_only_profile_binding():
    p=Path(__file__).parents[1]/'tools/launch_eigenpair_chain_lift.py';s=importlib.util.spec_from_file_location('lift_launcher',p);l=importlib.util.module_from_spec(s);s.loader.exec_module(l)
    l.ARGS=m.parser().parse_args(['--mode','target','--out',str(m.BASE/'attempt001-target/run'),'--profile','/profile','--profile-manifest','a'*64,'--profile-outer-manifest','b'*64]);l.OUT=m.BASE/'attempt001-target';l.LIMIT=40
    cmd=l.command();assert str(m.ROOT/'tools/probe_eigenpair_chain_lift.py')in cmd and '--proposal'not in cmd
