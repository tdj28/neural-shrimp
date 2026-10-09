from copy import deepcopy
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,eye
from neural_shrimp.rigorous_connection_frames import (framed_product,framed_tube,two_charts,anchored_at_parameter)

class Quadratic:
    n=1
    def step(self,x,h):return [x[0]*x[0]-1+h-1],arb_mat([[2*x[0]]]),[arb(1)]

def inputs():return {'rate':'1','initial':[['0'],['-1']],'anchor':{'rate':'1','phase_boxes':[[ball_json(arb(0))],[ball_json(arb(-1))]]}}
SETTINGS=dict(precision=256,halfwidth_exponent=-12,radius_exponent=-12,newton_steps=2,max_power=64)

def test_two_charts_common_endpoint_is_identified():
    r=two_charts(Quadratic(),inputs(),**SETTINGS)
    assert r['passed'] and all(r['gates'].values())
    a,b=r['charts'];assert a['exact_parameter_endpoints'][1]==b['exact_parameter_endpoints'][0]==b['anchor_parameter']
    assert all(b['anchor_inclusions'])


def test_same_rate_join_rejects_overlap_without_containment():
    r=two_charts(Quadratic(),inputs(),**SETTINGS);chart=r['charts'][1]
    # A broad candidate may overlap the chart but is not inside its uniqueness domain.
    a={'rate':chart['anchor_parameter'],'phase_boxes':[[ball_json(arb(0,1))],[ball_json(arb(-1,1))]]}
    with ctx.workprec(256):passed,_=anchored_at_parameter(chart,a)
    assert not passed
    a['rate']='3'
    with pytest.raises(ValueError):anchored_at_parameter(chart,a)


def test_cyclic_conjugation_and_order():
    with ctx.workprec(256):
        blocks=[arb_mat([[1,2],[0,1]]),arb_mat([[2,0],[3,1]]),arb_mat([[0,1],[1,0]])]
        frames=[arb_mat([[1,0],[0,2]]),arb_mat([[2,1],[0,1]]),arb_mat([[1,0],[1,1]])]
        m,changed,w=framed_product(blocks,frames)
        reference=frames[0].inv()*blocks[2]*blocks[1]*blocks[0]*frames[0]
        assert all(m[i,j].overlaps(reference[i,j]) for i in range(2) for j in range(2))
        assert len(w)==3


def test_singular_or_inexact_frames_fail():
    with pytest.raises((ValueError,ZeroDivisionError)):framed_product([eye(2)],[arb_mat([[1,0],[0,0]])])
    with pytest.raises(ValueError):framed_product([eye(1)],[arb_mat([[arb(1,1)]])])


def test_primitive_gate_not_replaced_by_frames():
    class Linear:
        n=1
        def step(self,x,h):return [x[0]/2+h-1],arb_mat([['1/2']]),[arb(1)]
    i=inputs();i['initial']=[['0'],['0']];i['anchor']['phase_boxes']=[[ball_json(arb(0))]]*2
    r=two_charts(Linear(),i,**SETTINGS)
    assert not r['passed'] and len(r['charts'])==1 and not r['gates']['connected_primitive_branch']


def test_completed_first_chart_is_emitted_before_second_failure(monkeypatch):
    import neural_shrimp.rigorous_connection_frames as module
    original=module.framed_tube;calls=0;saved=[]
    def interrupted(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise TimeoutError('injected second chart interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'framed_tube',interrupted)
    with pytest.raises(TimeoutError):
        module.two_charts(Quadratic(),inputs(),observe=lambda kind,row:saved.append((kind,row)),**SETTINGS)
    completed=[r for kind,r in saved if kind=='chart_completed']
    assert len(completed)==1 and completed[0]['chart']==0 and completed[0]['result']['passed']


def test_frame_proposal_failure_cannot_pass_attraction(monkeypatch):
    import neural_shrimp.rigorous_connection_frames as module
    def bad(_):raise ValueError('synthetic invalid frame')
    monkeypatch.setattr(module,'propose_frames',bad)
    r=module.two_charts(Quadratic(),inputs(),**SETTINGS)
    assert len(r['charts'])==2 and r['gates']['connected_primitive_branch']
    assert not r['gates']['uniform_attraction'] and not r['passed']


def test_compressed_chart_payload_identity_rejects_corruption(tmp_path):
    import importlib.util,json
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('framed_driver',Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_connection_frames.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    path=tmp_path/'stage-003-chart_completed.json.gz'
    payload=m.save(path,{'chart':0,'result':{'complete':True}})
    meta=path.with_name('stage-003-chart_completed-payload.json');m.save(meta,payload)
    assert m.read(path)['result']['complete']
    payload['uncompressed_sha256']='0'*64;meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError):m.read(path)
