import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_earlier_whole_stability.py';s=importlib.util.spec_from_file_location('earlier_whole',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_exact_whole_interval_only():
 assert m.single_coverage([m.ENDS],m.ENDS)==['25/128','27/128']
 for rows in ([],[['25/128','13/64']],[m.ENDS,m.ENDS]):
  with pytest.raises(ValueError):m.single_coverage(rows,m.ENDS)

def test_profile_never_reads_target(monkeypatch):
 marker=object();monkeypatch.setattr(m.previous.base,'input_for',lambda mode:marker);monkeypatch.setattr(m,'completed_adjacent',lambda:pytest.fail('target read'))
 assert m.input_for('profile')is marker

def fixture(monkeypatch):
 c={'exact_parameter_endpoints':m.ENDS,'normal':[1],'origin':[0],'endpoint':'full235'};n={'exact_parameter_endpoints':['27/128','29/128'],'normal':[1],'origin':[0],'anchor':'full235'}
 a={'chart':n,'model_inputs':'map','localized_join':{'old':'metadata'}};k={'model_inputs':'map'}
 monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'bound'});monkeypatch.setattr(m.previous.native,'compact_chart',lambda c:c)
 def join(x,y):m.require(x['endpoint']==y['anchor']and x['normal']==y['normal']and x['origin']==y['origin'],'full native join')
 monkeypatch.setattr(m.previous.native,'joined',join)
 return a,{'proved':'adjacent'},k,c,n

def test_literal141_to142_selection(monkeypatch):
 x=m.select(*fixture(monkeypatch));assert x['slices']==[m.ENDS]and x['selected_artifact']['global_chart']==141 and x['selected_artifact']['index']==11 and 'localized_join'not in x

@pytest.mark.parametrize('which',['map','endpoint','gauge','interval','adjacent'])
def test_changed_native_join_is_rejected(monkeypatch,which):
 a=fixture(monkeypatch)
 if which=='map':a[2]['model_inputs']='wrong'
 if which=='endpoint':a[3]['endpoint']='wrong'
 if which=='gauge':a[3]['normal']=[2]
 if which=='interval':a[3]['exact_parameter_endpoints']=['1','2']
 if which=='adjacent':a[0]['chart']={'wrong':'chart'}
 with pytest.raises(ValueError):m.select(*a)

def test_source_and_coverage_overrides_restore_after_failure(monkeypatch):
 saved=m.core.sources,m.core.input_for,m.core.coverage;monkeypatch.setattr(m,'input_for',lambda _: {'synthetic':True})
 def fail(*_):
  m.core.input_for('profile');assert m.core.coverage is m.single_coverage;raise OSError('durable failure')
 monkeypatch.setattr(m.core,'execute',fail)
 with pytest.raises(OSError):m.execute(m.RAW/'attempt001-profile/run','profile')
 assert (m.core.sources,m.core.input_for,m.core.coverage)==saved
 with pytest.raises(ValueError,match='exclusive'):m.execute(m.RAW/'other/run','profile')
