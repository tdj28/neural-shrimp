import importlib.util
from pathlib import Path
from fractions import Fraction
import pytest


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_weighted_reanchor.py'
    s=importlib.util.spec_from_file_location('weighted_reanchor',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def proposal():return {'proposal':{'radius_exponent':-21,'exact_weights':['1']*235,'exact_coordinate_radii':[str(Fraction(1,2**21))]*235}}


def test_exact_proposal_normalization():assert module().validate_proposal(proposal())==['1']*235

@pytest.mark.parametrize('change',['radius','zero','negative','non_dyadic','mismatch','missing'])
def test_corrupt_proposal_rejected(change):
    row=proposal();p=row['proposal']
    if change=='radius':p['radius_exponent']=-20
    if change=='zero':p['exact_weights'][0]='0'
    if change=='negative':p['exact_weights'][0]='-1'
    if change=='non_dyadic':p['exact_weights'][0]='1/3'
    if change=='mismatch':p['exact_coordinate_radii'][0]='1/4'
    if change=='missing':p['exact_weights'].pop()
    with pytest.raises(ValueError):module().validate_proposal(row)


def test_localized_anchor_not_failed_chart_preserved(monkeypatch):
    m=module();box={'state_box':['all native coordinates']};base={'weights':['old'],'anchor':box,'normal':['fixed'],'origin':['fixed'],'provenance':{}}
    monkeypatch.setattr(m.inherited,'prepare',lambda:base)
    monkeypatch.setattr(m,'proposed_weights',lambda:['new'])
    r=m.prepare();assert r['anchor'] is box and 'exact_parameter' not in r['anchor'] and r['weights']==['new']
    assert r['normal']==['fixed'] and r['origin']==['fixed']


def test_manifest_pin_before_saved_proposal_read(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'PROPOSAL',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):m.proposed_weights()
