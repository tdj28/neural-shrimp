import importlib.util
from pathlib import Path
from fractions import Fraction
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s2p18',ROOT/'tools/certify_s2_period18.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def fixture():
    x=np.arange(48,dtype=float).reshape(24,2)/16;y=np.arange(24,dtype=float)/8
    net=dict(d=2,width=3,bias=True,activation='tanh',in_scale=1.,out_scale=1.,loss_scale=.5,ridge=.001)
    exact={k:(M.dyadic(v) if k in ('in_scale','out_scale','loss_scale','ridge') else v) for k,v in net.items()}
    parent=dict(network=exact,x=[[M.dyadic(v) for v in row] for row in x],y=[M.dyadic(v) for v in y])
    row=dict(map_id='S2-28011-zoom',component=587,period=18,lr_hidden=71.337643833463,lr_output=.8210596690584651,certificate=dict(verified=True,attracting=True,primitive_period=18,orbit=np.zeros((18,13)).tolist()))
    plan=dict(seed=28011,system='S2',network=net,data_sha256=dict(x=M.shape_hash(x),y=M.shape_hash(y)))
    return parent,row,plan

def test_consumed_data_adapter_preserves_exact_rates_and_arrays():
    parent,row,plan=fixture();out=M.convert(parent,row,plan)
    assert out['x'] is parent['x'] and out['y'] is parent['y'] and out['period']==18
    assert Fraction(out['lr_hidden'])==Fraction(row['lr_hidden']) and len(out['approximation'])==13


def test_data_change_rejected_without_regeneration():
    parent,row,plan=fixture();parent['x'][0][0]='1/2'
    with pytest.raises(ValueError,match='training hash'):M.convert(parent,row,plan)


def test_whole_box_proximity_is_not_seed_membership():
    inputs=dict(approximation=['0']*13);box=[dict(mid=[1,-30],rad=[1,-80])]*13
    out=M.proximity(inputs,box);assert out['passed'];assert Fraction(out['normalized_whole_box_distance_upper'])==Fraction(2)**-30+Fraction(2)**-80
    assert Fraction(2)**-30>Fraction(2)**-80


def test_whole_box_threshold_fails_exactly():
    inputs=dict(approximation=['0']*13);box=[dict(mid=[1,-20],rad=[1,-80])]*13
    assert not M.proximity(inputs,box)['passed']
