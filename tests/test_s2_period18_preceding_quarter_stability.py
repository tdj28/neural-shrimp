import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_preceding_quarter_stability.py'
s=importlib.util.spec_from_file_location('preceding_quarters',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_exact_preceding_partition():
    assert m.base.partition(['33/128','17/64'])==[['135/512','17/64'],['67/256','135/512'],['133/512','67/256'],['33/128','133/512']]

def test_profile_has_no_adjacent_target_read(monkeypatch):
    marker=object();monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:marker)
    monkeypatch.setattr(m,'completed_adjacent',lambda:pytest.fail('target read'))
    assert m.input_for('profile')is marker

def test_target_requires_completed_adjacent(monkeypatch):
    monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:{})
    def stop():raise ValueError('missing adjacent')
    monkeypatch.setattr(m,'completed_adjacent',stop)
    with pytest.raises(ValueError,match='missing adjacent'):m.input_for('target')

def test_execute_restores_adapter_on_failure(monkeypatch):
    old=m.base.sources,m.base.input_for
    def fail(*args):
        assert m.base.sources is m.sources and m.base.input_for is m.input_for
        raise ValueError('failure')
    monkeypatch.setattr(m.base,'execute',fail)
    with pytest.raises(ValueError,match='failure'):m.execute(Path('/unused'),'profile')
    assert (m.base.sources,m.base.input_for)==old

def fixture_select(monkeypatch):
    first={'exact_parameter_endpoints':['33/128','17/64'],'normal':[1],'origin':[0],'endpoint':'full235'}
    nextchart={'normal':[1],'origin':[0],'anchor':'full235'}
    value={'chart':nextchart,'model_inputs':'fixedmap'}
    adjacent=dict(value);prepared={'model_inputs':'fixedmap','chart':'failed146'}
    monkeypatch.setattr(m.saved,'target_pair',lambda pair,inp:'failed146')
    monkeypatch.setattr(m.saved,'read',lambda p:{})
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'bound'})
    monkeypatch.setattr(m.native,'compact_chart',lambda x:x)
    def join(a,b):m.require(a['endpoint']==b['anchor'],'full native endpoint')
    monkeypatch.setattr(m.native,'joined',join)
    return value,adjacent,{'source':'completed146'},prepared,{'charts':[first,'failed146']}

def test_select_complete_native_predecessor(monkeypatch):
    args=fixture_select(monkeypatch);value=m.select(*args)
    assert value['selected_artifact']['global_chart']==145
    assert value['coverage_endpoints']==['33/128','17/64']
    assert value['coverage_predecessor']=={'source':'completed146'}

@pytest.mark.parametrize('mutation',['map','endpoint','gauge','endpoints'])
def test_select_rejects_corruption(monkeypatch,mutation):
    args=fixture_select(monkeypatch)
    if mutation=='map':args[3]['model_inputs']='other'
    elif mutation=='endpoint':args[4]['charts'][0]['endpoint']='different'
    elif mutation=='gauge':args[4]['charts'][0]['normal']=[2]
    else:args[4]['charts'][0]['exact_parameter_endpoints']=['1/4','17/64']
    with pytest.raises(ValueError):m.select(*args)
