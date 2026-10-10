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

def admission(index=1):
 return {'index':index,'first_global_chart':22+4*(index-1),'selected':[{}]*(1 if index==13 else 4),'prior_root':str(R.BASE/('unit000-attempt002'if index==1 else f'unit{index-1:03d}-attempt001')),'prior_endpoint':'run/endpoint.json','prior_manifest':'a'*64,'outcome_passed':index!=13,'producer_exit':2 if index==13 else 0}

def test_fixed_complete_units_and_disclosed_partial_unit():
 for i in [1,2,12,13]:assert R.admit(R.BASE/f'unit{i:03d}-attempt001/run',R.ROOT/f'docs/plans/eigenpair-remaining-unit{i:03d}.json',admission(i))==i

@pytest.mark.parametrize('key,value',[('index',14),('index',True),('first_global_chart',99),('prior_root','/wrong'),('prior_manifest',None),('selected',[{}]),('outcome_passed',False),('producer_exit',2)])
def test_unfrozen_or_mismatched_next_unit_fails(key,value):
 d=admission();d[key]=value
 with pytest.raises(ValueError):R.admit(R.BASE/'unit001-attempt001/run',R.ROOT/'docs/plans/eigenpair-remaining-unit001.json',d)


@pytest.mark.parametrize('i,first,count',[(14,71,1),(15,72,2),(16,74,4),(21,94,4),(22,98,4),(23,102,4),(24,106,2)])
def test_tail_exact_complete_units(i,first,count):
 d=admission(i);d.update(first_global_chart=first,selected=[{}]*count)
 assert R.admit(R.BASE/f'unit{i:03d}-attempt001/run',R.ROOT/f'docs/plans/eigenpair-remaining-unit{i:03d}.json',d)==i
 d['selected']=d['selected'][:-1]
 with pytest.raises(ValueError):R.admit(R.BASE/f'unit{i:03d}-attempt001/run',R.ROOT/f'docs/plans/eigenpair-remaining-unit{i:03d}.json',d)

def test_direct_tail_preserves_terminal_status_and_complete_count():
 d={'producer_status':'one_repaired_partial_chart'}
 R.producer_selection({'status':'one_repaired_partial_chart'},{'passed':True,'charts':[{}]},d,14)
 for selected in [{'passed':False,'charts':[{}]},{'passed':True,'charts':[]}]:
  with pytest.raises(ValueError):R.producer_selection({'status':'one_repaired_partial_chart'},selected,d,14)
 with pytest.raises(ValueError):R.producer_selection({'status':'four_joined_charts'},{'passed':True,'charts':[{}]},d,14)

def test_event_terminal_is_not_ordinary_completion():
 for i,status in [(21,'qualified_event_stop'),(24,'rigorous_chain_membership')]:
  rows=[{'record':{'index':k}}for k in range(R.unit_shape(i)[1])]
  spec={'selected':rows,'producer_status':status};done={'status':status,'progress':{'accepted':[r['record']for r in rows]}}
  R.producer_selection(done,None,spec,i)
  done['status']='four_joined_charts'
  with pytest.raises(ValueError):R.producer_selection(done,None,spec,i)
