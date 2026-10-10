import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_two_predecessor_stability.py'
s=importlib.util.spec_from_file_location('two_predecessors',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_only_fixed_immediate_paths():
    assert m.paths(13)==(m.FIRST,m.RAW/'unit001/run')
    assert m.paths(12)==(m.RAW/'unit001/run',m.RAW/'unit002/run')
    for index in (14,11,True,None):
        with pytest.raises(ValueError):m.paths(index)

def test_profile_never_checks_target_predecessor(monkeypatch):
    marker=object();monkeypatch.setattr(m.base,'input_for',lambda mode:marker)
    monkeypatch.setattr(m,'completed',lambda *args:pytest.fail('target read'))
    assert m.input_for('profile')is marker

def test_wrong_predecessor_path_rejected_before_file_read():
    with pytest.raises(ValueError,match='fixed immediate'):m.completed(12,m.FIRST,'unused','unused')

def fixture(monkeypatch,index):
    chart={'exact_parameter_endpoints':m.ENDS[index],'normal':[1],'origin':[0],'endpoint':'full235'}
    nxt={'anchor':'full235','normal':[1],'origin':[0]}
    value={'model_inputs':'map','localized_join':{'old':'join'}};adj={'chart':nxt,'model_inputs':'map'};known={'model_inputs':'map'}
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'bound'});monkeypatch.setattr(m.native,'compact_chart',lambda c:c)
    def join(a,b):
        m.require(a['endpoint']==b['anchor'],'complete native endpoint');m.require(a['normal']==b['normal']and a['origin']==b['origin'],'same gauge')
    monkeypatch.setattr(m.native,'joined',join)
    return index,value,adj,{'previous':'complete'},known,chart,nxt

@pytest.mark.parametrize('index',[13,12])
def test_exact_eight_piece_ordinary_native_join(monkeypatch,index):
    v=m.select(*fixture(monkeypatch,index));assert v['selected_artifact']['global_chart']==130+index
    assert len(v['slices'])==8 and 'localized_join'not in v
    assert m.core.coverage(v['slices'],m.ENDS[index])==m.ENDS[index]

@pytest.mark.parametrize('which',['map','endpoint','gauge','interval','adjacent'])
def test_corruptions_fail_closed(monkeypatch,which):
    a=fixture(monkeypatch,13)
    if which=='map':a[4]['model_inputs']='other'
    elif which=='endpoint':a[5]['endpoint']='other'
    elif which=='gauge':a[5]['normal']=[2]
    elif which=='interval':a[5]['exact_parameter_endpoints']=['1/4','31/128']
    else:a[2]['chart']={'different':'native predecessor'}
    with pytest.raises(ValueError):m.select(*a)
