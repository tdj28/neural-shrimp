import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
from flint import arb
from neural_shrimp.rigorous_orbits import vector_json


def module():
    path=Path(__file__).parents[1]/'tools/certify_s2_period18_reanchor.py'
    spec=importlib.util.spec_from_file_location('reanchor_driver',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def objects():
    localized={'passed':True,'scientific_claim':True,'parent_gates':{'root':True},'statement':{'period':18,'phase_dimension':13},
      'iterations':3,'precision':256,'final_state_box':vector_json([arb(0,arb('1e-15'))]*234+[arb(75,arb('1e-15'))])}
    old={'model_inputs':{'exact':'unchanged'},'provenance':{'original_anchor':{'fixed':True}}}
    proposal={'normal':['0']*234+['1'],'origin':['0']*234+['74']}
    return localized,old,proposal


def test_complete_anchor_correlation_and_model_preserved(monkeypatch):
    m=module();monkeypatch.setattr(m.io,'prepare_weights',lambda:['1']*234)
    localized,old,proposal=objects();r=m.inherited_inputs(localized,old,proposal)
    assert r['anchor']=={'state_box':localized['final_state_box']}
    assert 'exact_parameter' not in r['anchor']
    assert r['origin'][-1]=='74' and r['initial'][-1]=='75'
    assert r['model_inputs']==old['model_inputs'] and r['normal']==proposal['normal']

@pytest.mark.parametrize('field,value',[('passed',False),('scientific_claim',False),('iterations',2),('precision',128)])
def test_incomplete_or_profile_parent_rejected(field,value,monkeypatch):
    m=module();monkeypatch.setattr(m.io,'prepare_weights',lambda:['1']*234)
    localized,old,proposal=objects();localized[field]=value
    with pytest.raises(ValueError):m.inherited_inputs(localized,old,proposal)

@pytest.mark.parametrize('kind',['missing_coordinate','zero_normal','non_dyadic'])
def test_bad_gauge_or_truncated_anchor_rejected(kind,monkeypatch):
    m=module();monkeypatch.setattr(m.io,'prepare_weights',lambda:['1']*234)
    localized,old,proposal=objects()
    if kind=='missing_coordinate':localized['final_state_box'].pop()
    elif kind=='zero_normal':proposal['normal']=['0']*235
    else:proposal['normal'][0]='1/3'
    with pytest.raises(ValueError):m.inherited_inputs(localized,old,proposal)


def test_profile_gate_mismatch_rejected_before_science_read(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):module().profile_gate(tmp_path,'0'*64)


def test_parent_manifest_checked_before_parent_result(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'PARENT',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):m.prepare()
