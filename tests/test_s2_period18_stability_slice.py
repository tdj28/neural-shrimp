import importlib.util
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_stability_slice.py'
    spec=importlib.util.spec_from_file_location('fresh_slice_driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(monkeypatch,m):
    rt={'mode':'profile','source_sha256':{},'python':m.platform.python_version(),'python_flint':m.io.flint.__version__,'flint':m.io.flint.__FLINT_VERSION__,
      'numpy':m.io.numpy.__version__,'scipy':m.io.scipy.__version__,'precision':256,'map_cap':162,'unit_derivative_cap':324,'mixed_call_cap':18,'internal_seconds':30,'tick_cap':800,'threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    out={'exit_code':0,'status':'synthetic_profile_completed','adapter_counts':{'map_attempted':162,'map_returned':162,'unit_attempted':324,'unit_returned':324,'map_cap':162,'unit_cap':324},
      'mixed_counts':{'attempted':18,'returned':18,'cap':18},'completed_intersections':6,'completed_physical_phases':18,'completed_products':18,'stages':85,'result_status':'complete_slice_bounds','passed':True,'elapsed_seconds':1,'max_rss_kib':100}
    launch={'mode':'profile','source_sha256':{},'exit_code':0,'owned_live_pids':[],'elapsed_seconds':2,'max_group_rss_kib':200}
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'pin'});monkeypatch.setattr(m.io,'verify_inventory',lambda *a:None)
    monkeypatch.setattr(m,'sources',lambda:{});monkeypatch.setattr(m.io,'total',lambda p:100)
    monkeypatch.setattr(m.io,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':rt,'outcome.json':out,'launcher-result.json':launch}[p.name])
    return rt,out,launch


@pytest.mark.parametrize('key,value',[('elapsed_seconds',0),('elapsed_seconds',float('nan')),('elapsed_seconds',20),('max_rss_kib',0),('max_rss_kib',1024**2),
 ('completed_intersections',3),('completed_physical_phases',17),('completed_products',17),('stages',84),('passed',False),('result_status','inconclusive_uniform_tangent')])
def test_profile_cannot_admit_missing_work_or_bad_resources(tmp_path,monkeypatch,key,value):
    m=driver();rt,out,launch=fixture(monkeypatch,m);assert m.profile_gate(tmp_path,'pin','pin')['projected_seconds']==14
    out[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')


@pytest.mark.parametrize('key,value',[('exit_code',1),('owned_live_pids',[12]),('max_group_rss_kib',0),('elapsed_seconds',float('inf')),('source_sha256',{'x':'changed'})])
def test_outer_profile_identity_and_owned_exit(tmp_path,monkeypatch,key,value):
    m=driver();rt,out,launch=fixture(monkeypatch,m);launch[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')


def test_exact_libraries_and_both_manifests(tmp_path,monkeypatch):
    m=driver();rt,out,launch=fixture(monkeypatch,m)
    for a,b in [('wrong','pin'),('pin','wrong')]:
        with pytest.raises(ValueError):m.profile_gate(tmp_path,a,b)
    rt['flint']='changed'
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')
    assert 'src/neural_shrimp/models.py' in m.SOURCES and 'src/neural_shrimp/controls.py' in m.SOURCES


def test_exact_doubled_target_and_no_hidden_search(monkeypatch):
    m=driver();parent={'chart':{'exact_parameter_endpoints':['11/32','45/128']}}
    monkeypatch.setattr(m.io.io,'partial_input',lambda:parent)
    assert m.input_for('target')['exact_endpoints']==['359/1024','45/128']
    parent['chart']['exact_parameter_endpoints']=['0','1']
    with pytest.raises(ValueError):m.input_for('target')


def test_storage_error_type_is_outside_numerical_catch():
    m=driver();assert issubclass(m.DurabilityError,RuntimeError) and not issubclass(m.DurabilityError,ValueError)
