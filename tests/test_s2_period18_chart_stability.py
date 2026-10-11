import importlib.util
from pathlib import Path
from fractions import Fraction
import pytest

def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_chart_stability.py';s=importlib.util.spec_from_file_location('chart_stability_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_exact_partition_reuses_only_last_slice():
    m=driver();rows=m.partition();assert len(rows)==7 and rows[0]==['179/512','359/1024'] and rows[-1]==['11/32','353/1024']
    assert m.coverage(rows,['359/1024','45/128'])==['11/32','45/128']
    with pytest.raises(ValueError):m.coverage(rows[:-1]+[rows[0]],['359/1024','45/128'])
    with pytest.raises(ValueError):m.coverage(rows,['719/2048','45/128'])

def test_stream_retains_failed_slice_then_stops():
    m=driver();called=[];retained=[];inp={'slices':m.partition(),'reused':{'exact_endpoints':['359/1024','45/128']}}
    def construct(i):called.append(i);return None
    result=m.stream(inp,'target',construct,lambda i,*a:{'passed':i==0,'status':'done'},lambda i,r:retained.append(i))
    assert called==retained==[0,1] and not result['passed'] and result['complete_coverage'] is None

def test_retention_failure_never_advances():
    m=driver();called=[]
    def retain(*a):raise OSError('full')
    with pytest.raises(OSError):m.stream({'slices':m.partition()},'target',lambda i:called.append(i),lambda *a:{'passed':True,'status':'done'},retain)
    assert called==[0]

def test_complete_stream_exactcoverage_and_no_eighth_acquisition():
    m=driver();called=[];inp={'slices':m.partition(),'reused':{'exact_endpoints':['359/1024','45/128']}}
    result=m.stream(inp,'target',lambda i:called.append(i),lambda *a:{'passed':True,'status':'done'},lambda *a:None)
    assert called==list(range(7)) and result['complete_coverage']==['11/32','45/128']

def fixture(monkeypatch,m):
    rt={'mode':'profile','source_sha256':{},'python':m.platform.python_version(),'python_flint':m.io.flint.__version__,'flint':m.io.flint.__FLINT_VERSION__,'numpy':m.io.numpy.__version__,'scipy':m.io.scipy.__version__,'precision':256,'internal_seconds':30,'per_slice_C1_cap':162,'per_slice_mixed_cap':36,'tick_cap':6400,'threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    out={'exit_code':0,'passed':True,'completed_slices':1,'stages':161,'counts':{'map_attempted':162,'map_returned':162,'unit_attempted':324,'unit_returned':324,'mixed_attempted':36,'mixed_returned':36},'elapsed_seconds':1,'max_rss_kib':100}
    launch={'mode':'profile','source_sha256':{},'exit_code':0,'owned_live_pids':[],'elapsed_seconds':2,'max_group_rss_kib':200}
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'pin'});monkeypatch.setattr(m.io,'verify_inventory',lambda *a:None);monkeypatch.setattr(m,'sources',lambda:{});monkeypatch.setattr(m.io,'total',lambda p:100)
    monkeypatch.setattr(m.io,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':rt,'outcome.json':out,'launcher-result.json':launch}[p.name]);return rt,out,launch

@pytest.mark.parametrize('key,value',[('elapsed_seconds',0),('elapsed_seconds',float('nan')),('elapsed_seconds',14),('max_rss_kib',1024**2),('stages',160),('completed_slices',0),('passed',False),('counts',{})])
def test_profile_full_work_and_resource_gate(tmp_path,monkeypatch,key,value):
    m=driver();rt,out,launch=fixture(monkeypatch,m);assert m.profile_gate(tmp_path,'pin','pin')['projected_seconds']==36
    out[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')

@pytest.mark.parametrize('key,value',[('exit_code',1),('owned_live_pids',[12]),('source_sha256',{'x':'changed'}),('elapsed_seconds',float('inf'))])
def test_launcher_admission(tmp_path,monkeypatch,key,value):
    m=driver();rt,out,launch=fixture(monkeypatch,m);launch[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')

def test_both_pins_and_libraries(tmp_path,monkeypatch):
    m=driver();rt,out,launch=fixture(monkeypatch,m)
    for a,b in [('bad','pin'),('pin','bad')]:
        with pytest.raises(ValueError):m.profile_gate(tmp_path,a,b)
    rt['flint']='changed'
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')
