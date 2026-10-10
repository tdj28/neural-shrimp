import sys,gzip,json,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import review_eigenpair_remaining as R

def test_only_immediately_preceding_attempt_events_are_used(tmp_path):
 events=tmp_path/'events';events.mkdir()
 def save(name,row):
  with gzip.open(events/name,'wt')as f:json.dump(row,f)
 save('0000-phase_return.json.gz',{'old':True});save('0001-chart_return.json.gz',{'passed':False})
 save('0002-march_diagnostics.json.gz',{});save('0003-phase_return.json.gz',{'new':True});save('0004-path_assembly_return.json.gz',{})
 save('0005-chart_return.json.gz',{'passed':True});save('0006-phase_return.json.gz',{'later':True})
 found,attempts=R.event_group(tmp_path,'events/0005-chart_return.json.gz')
 assert found==[('phase_return',{'new':True}),('path_assembly_return',{})]
 assert attempts==['events/0001-chart_return.json.gz','events/0005-chart_return.json.gz']
 with pytest.raises(ValueError):R.event_group(tmp_path,'events/missing-chart_return.json.gz')
 with pytest.raises(ValueError):R.event_group(tmp_path,'outside-chart_return.json.gz')

def fixture():
 previous={'normal':['0','1'],'origin':['0','0'],'period':18,'dimension':13,'sigma':1,'shooting_dimension':470,'parameter_endpoints':['0','1'],'endpoint_root_boxes':[[],['all470']]}
 chart={k:v for k,v in previous.items()};chart['parameter_endpoints']=['1','2'];return previous,chart

def test_native_join_preserves_entire_endpoint():
 p,c=fixture();assert R.joined(p,c)=={'box':['all470'],'exact_parameter':'1'}

@pytest.mark.parametrize('key,value',[('normal',['1','0']),('origin',['1','0']),('parameter_endpoints',['2','3']),('sigma',-1),('dimension',12)])
def test_corrupted_join_fails(key,value):
 p,c=fixture();c[key]=value
 with pytest.raises(ValueError):R.joined(p,c)

def test_manifest_requires_exact_member_set_and_immutable_files(tmp_path):
 p=tmp_path/'a';p.write_text('data');p.chmod(0o444)
 m=tmp_path/'manifest.json';m.write_text(json.dumps({'files':{'a':R.entry(p)}}));m.chmod(0o444);pin=R.entry(m)['sha256'];R.verify(tmp_path,pin)
 other=tmp_path/'unexpected';other.write_text('extra')
 with pytest.raises(ValueError,match='complete'):R.verify(tmp_path,pin)
 other.unlink();p.chmod(0o644)
 with pytest.raises(ValueError,match='immutable'):R.verify(tmp_path,pin)


def test_full_eigenpair_map_binding_rejects_old_orbit_envelope_and_changed_rates():
 known={'schema':'map','network':'native13','x':[[1]],'y':[1],'lr_hidden':'2','lr_output':'1'}
 R.same_map(dict(known),known)
 for extra in [dict(known,approximation=[0]),dict(known,lr_hidden='3'),dict(known,x=[[2]]),dict(known,network='other')]:
  with pytest.raises(ValueError):R.same_map(extra,known)
