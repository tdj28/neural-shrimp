import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_localized_predecessor_stability.py'
s=importlib.util.spec_from_file_location('localized_predecessor',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_profile_has_no_target_read(monkeypatch):
    marker=object();monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:marker)
    monkeypatch.setattr(m,'completed_adjacent',lambda:pytest.fail('target read'))
    assert m.input_for('profile')is marker

def test_execute_restores_core_on_failure(monkeypatch):
    old=m.core.sources,m.core.input_for
    def fail(*args):
        assert m.core.sources is m.sources and m.core.input_for is m.input_for
        raise ValueError('failure')
    monkeypatch.setattr(m.core,'execute',fail)
    with pytest.raises(ValueError,match='failure'):m.execute(Path('/unused'),'profile')
    assert (m.core.sources,m.core.input_for)==old

def fixture_select(monkeypatch):
    chart={'exact_parameter_endpoints':['31/128','33/128'],'normal':[1],'origin':[0]}
    anchor={'state_box':['localized full235'],'exact_parameter':'33/128'}
    nextchart={'normal':[1],'origin':[0],'anchor':anchor,'exact_parameter_endpoints':['33/128','17/64']}
    value={'chart':nextchart,'model_inputs':'fixedmap'}
    adjacent=dict(value);known={'model_inputs':'fixedmap','previous':chart}
    rechart={'model_inputs':'fixedmap','anchor':anchor,'normal':[1],'origin':[0]}
    old=dict(known)
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'bound'})
    monkeypatch.setattr(m.native,'compact_chart',lambda x:x)
    return value,adjacent,{'source':'completed145'},known,rechart,chart,old

def test_select_authenticated_localized_join(monkeypatch):
    value=m.select(*fixture_select(monkeypatch))
    assert value['selected_artifact']['global_chart']==144
    assert len(value['slices'])==8
    assert value['slices'][0]==['131/512','33/128'] and value['slices'][-1]==['31/128','125/512']

@pytest.mark.parametrize('mutation',['map','parent','anchor','gauge','parameter'])
def test_select_rejects_corruption(monkeypatch,mutation):
    args=fixture_select(monkeypatch)
    if mutation=='map':args[4]['model_inputs']='other'
    elif mutation=='parent':args[6]['previous']={'wrong':'parent'}
    elif mutation=='anchor':args[4]['anchor']={'state_box':['other full235'],'exact_parameter':'33/128'}
    elif mutation=='gauge':args[4]['normal']=[2]
    else:args[4]['anchor']['exact_parameter']='1/4'
    with pytest.raises(ValueError):m.select(*args)
