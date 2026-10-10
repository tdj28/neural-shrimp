import copy,importlib.util
from pathlib import Path
import pytest
from neural_shrimp.rigorous_adaptive_arclength import stream_radii
from test_rigorous_arclength_batch import parent
from test_rigorous_arclength_connection import Fold
from test_rigorous_arclength_march import fake_chain


def driver():
    spec=importlib.util.spec_from_file_location('radiidriver',Path(__file__).parents[1]/'tools/certify_s2_period18_arclength_radii.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_changed_box_reacquired_predecessor_not_relabelled():
    old=parent();original=copy.deepcopy(old);saved=[]
    r=stream_radii(Fold(),Fold(),old,new_weights=['1/2','1'],new_radius_exponent=-21,count=3,
       emit_chart=lambda i,c:saved.append(copy.deepcopy(c)))
    assert r['passed'] and old==original
    for c in saved:
        assert c['settings']['radius_exponent']==-21 and c['exact_weights']==['1/2','1']
        assert c['anchor']['state_box']==old['endpoint_root_enclosures'][1]
        old=c
    assert saved[0]['anchor']['state_box']==original['endpoint_root_enclosures'][1]


def test_storage_failure_stops_before_next_chart_and_keeps_parent():
    old=parent();before=copy.deepcopy(old);calls=[]
    def fail(i,c):calls.append(i);raise OSError('synthetic full disk')
    with pytest.raises(OSError):stream_radii(Fold(),Fold(),old,new_weights=['1','1'],new_radius_exponent=-21,count=3,emit_chart=fail)
    assert calls==[0] and old==before


@pytest.mark.parametrize('weights,rad',[(['0','1'],-21),(['-1','1'],-21),(['2','1'],-21),(['1/3','1'],-21),(['1'],-21),(['1','1'],0),(['1','1'],True)])
def test_invalid_box_proposals(weights,rad):
    with pytest.raises(ValueError):stream_radii(Fold(),Fold(),parent(),new_weights=weights,new_radius_exponent=rad)


def prefix():
    base,rows,charts=fake_chain();charts=charts[:13];rows=rows[:13]
    charts[-1]['passed']=False;charts[-1]['gates']['uniform_root']=False
    for row,c in zip(rows,charts):row.update(passed=c['passed'],gates=c['gates'])
    return base,rows,charts


def run_prefix(base,rows,charts):
    return driver().validate_prefix(base,rows,lambda row:charts[row['index']],expected_count=13,allow_last_failure=True,
       expected_settings=driver().SETTINGS,expected_weights=['fixed'])


def test_only_qualified_prefix_supplies_endpoint():
    base,rows,charts=prefix();last,count=run_prefix(base,rows,charts)
    assert count==12 and last['endpoint_root_enclosures']==charts[11]['endpoint_root_enclosures']
    assert last['endpoint_root_enclosures']!=charts[12]['endpoint_root_enclosures']


@pytest.mark.parametrize('mutate',['early_failure','bad_tail','tail_anchor','wrong_gauge','bad_inclusion','short','mirror','settings','weights'])
def test_partial_prefix_contradictions_rejected(mutate):
    base,rows,charts=prefix()
    if mutate=='early_failure':charts[4]['passed']=False;rows[4]['passed']=False
    if mutate=='bad_tail':charts[-1]['gates']['primitive_period']=False
    if mutate=='tail_anchor':charts[-1]['anchor']['state_box']=['wrong']
    if mutate=='wrong_gauge':charts[4]['normal']=['wrong']
    if mutate=='bad_inclusion':charts[4]['coordinate_inclusions'][7]=False
    if mutate=='short':rows.pop()
    if mutate=='mirror':rows[4]['passed']=False
    if mutate=='settings':charts[4]['settings']['radius_exponent']=-21
    if mutate=='weights':charts[4]['exact_weights']=['wrong']
    with pytest.raises(ValueError):run_prefix(base,rows,charts)


def test_complete_later_adapted_batch():
    base,rows,charts=fake_chain();m=driver()
    for row,c in zip(rows,charts):row['gates']=c['gates']
    last,count=m.validate_prefix(base,rows,lambda row:charts[row['index']],expected_count=16,allow_last_failure=False,expected_settings=m.SETTINGS,expected_weights=['fixed'])
    assert count==16 and last['passed']


def test_all_new_files_bound_and_proposal_pin_required(tmp_path,monkeypatch):
    m=driver()
    for name in ('src/neural_shrimp/rigorous_adaptive_arclength.py','tools/certify_s2_period18_arclength_radii.py','tests/test_rigorous_adaptive_arclength.py'):
        assert name in m.SOURCES
    monkeypatch.setattr(m,'DIAG',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='proposal manifest'):m.prepare_radii()
