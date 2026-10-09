from copy import deepcopy
from fractions import Fraction
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,matrix_json
from neural_shrimp.rigorous_metric_slices import partition,derivative_slice


def chart():
    return {'gates':{'uniform_root':True},'statement':{'period':18},'exact_parameter_endpoints':['1','2'],'parameter_center':'3/2',
        'center':[[ball_json(arb(0)) for _ in range(13)] for _ in range(18)],
        'predictor_slope':matrix_json(arb_mat([[1] for _ in range(234)])),
        'image_correction':matrix_json(arb_mat([[arb(0,'1/100')]]*234))}


class Model:
    n=13
    def __init__(self):self.calls=[]
    def step(self,x,h):
        self.calls.append((x,h));return x,arb_mat(13,13),x


def test_exact_closed_coverage_and_no_implicit_refinement():
    rows=partition(chart());assert len(rows)==16 and rows[0]==('1','17/16') and rows[-1]==('31/16','2')
    assert sum(Fraction(b)-Fraction(a) for a,b in rows)==1
    with pytest.raises(ValueError):partition(chart(),32)


def test_floor_keeps_proved_error_and_restricts_only_parameter():
    with ctx.workprec(256):
        m=Model();_,domain=derivative_slice(m,chart(),'3/2','3/2')
        assert len(m.calls)==18 and domain['floor_only']
        assert m.calls[0][1].is_exact()
        assert m.calls[0][0][0].contains(arb('1/100')) and not m.calls[0][0][0].is_exact()


def test_outside_or_unproved_parent_rejected_before_calls():
    m=Model()
    with pytest.raises(ValueError):derivative_slice(m,chart(),'0','1')
    c=deepcopy(chart());c['gates']['uniform_root']=False
    with pytest.raises(ValueError):derivative_slice(m,c,'1','2')
    assert not m.calls


def test_floor_gate_requires_both_native_parameters_and_success(monkeypatch,tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('slice_driver',Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_sliced_metric.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    parent={'charts':[{'parameter_center':'1'},{'parameter_center':'2'}]}
    rows=[{'chart':j,'kind':'floor','passed':True,'exact_endpoints':[str(j+1)]*2,'file':f'unit{j}.json.gz','identity':{'sha256':'pin'}} for j in range(2)]
    data={'manifest.json':{'files':{}},'runtime-start.json':{'source_sha256':{},'source_commit':'source','mode':'floor'},
        'outcome.json':{'status':'floors_certified'},'result.json':{'passed':True,'floors':[True,True],'parent_certificate_sha256':m.PARENT_CERTIFICATE,'units':rows}}
    for j in range(2):data[f'unit{j}.json.gz']={'record':{'proof':{'passed':True},'domain':{'exact_endpoints':[str(j+1)]*2}}}
    monkeypatch.setattr(m,'read',lambda p:data[p.name]);monkeypatch.setattr(m,'entry',lambda p:{'sha256':'pin'})
    monkeypatch.setattr(m,'verify_inventory',lambda *args:None);monkeypatch.setattr(m,'sources',lambda:{})
    assert m.floor_gate(tmp_path,'pin',parent)['passed']
    rows[1]['exact_endpoints']=['1','1']
    with pytest.raises(ValueError):m.floor_gate(tmp_path,'pin',parent)
    rows[1]['exact_endpoints']=['2','2'];data['result.json']['floors']=[True,False]
    with pytest.raises(ValueError):m.floor_gate(tmp_path,'pin',parent)
