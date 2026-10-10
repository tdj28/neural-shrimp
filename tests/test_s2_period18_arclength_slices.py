import importlib.util
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_arclength_slices.py'
    s=importlib.util.spec_from_file_location('arc_slice_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_fixed_floor_and_right_sixteenth():
    m=driver();c={'exact_parameter_endpoints':['11/32','45/128']}
    assert m.domains(c,'target')==[('floor','45/128','45/128'),('probe','719/2048','45/128')]
    assert m.domains(c,'profile')==[('probe','719/2048','45/128')]


def profile_fixture(monkeypatch,m):
    runtime={'mode':'profile','source_sha256':{},'python':m.platform.python_version(),'python_flint':m.flint.__version__,
      'flint':m.flint.__FLINT_VERSION__,'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'precision':256,'map_cap':90,
      'unit_derivative_cap':180,'internal_seconds':20,'tick_cap':400,'threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    result={'exit_code':0,'status':'synthetic_profile_completed','adapter_counts':{'map_attempted':90,'map_returned':90,'unit_attempted':180,'unit_returned':180,'map_cap':90,'unit_cap':180},
      'elapsed_seconds':1,'max_rss_kib':100,'completed_intersections':3,'completed_cases':1,'completed_physical_phases':18,
      'cases':[{'role':'probe','passed':True,'point_passed':True,'interval_phases_passed':18}]}
    monkeypatch.setattr(m,'entry',lambda p:{'sha256':'pin'});monkeypatch.setattr(m,'verify_inventory',lambda *x:None)
    monkeypatch.setattr(m,'sources',lambda:{});monkeypatch.setattr(m,'total',lambda p:100)
    monkeypatch.setattr(m,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':runtime,'outcome.json':result}[p.name])
    return runtime,result


@pytest.mark.parametrize('key,value',[('elapsed_seconds',-1),('elapsed_seconds',float('nan')),('elapsed_seconds',20),
    ('max_rss_kib',-1),('max_rss_kib',float('inf')),('max_rss_kib',1024**2),
    ('completed_intersections',2),('completed_physical_phases',17),('completed_cases',0),('cases',[])])
def test_profile_rejects_resource_or_coverage_contradictions(tmp_path,monkeypatch,key,value):
    m=driver();rt,r=profile_fixture(monkeypatch,m)
    assert m.profile_gate(tmp_path,'pin')['projected_seconds']==14
    r[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin')


def test_profile_library_and_manifest_binding(tmp_path,monkeypatch):
    m=driver();rt,r=profile_fixture(monkeypatch,m)
    with pytest.raises(ValueError,match='manifest'):m.profile_gate(tmp_path,'wrong')
    rt['python_flint']='changed'
    with pytest.raises(ValueError,match='library'):m.profile_gate(tmp_path,'pin')


def test_exact_full_chart_required_after_parent_authentication(monkeypatch):
    m=driver();monkeypatch.setattr(m.parent,'partial_input',lambda:{'previous':{}})
    monkeypatch.setattr(m,'read',lambda p:{})
    monkeypatch.setattr(m,'entry',lambda p:{'sha256':'wrong'})
    with pytest.raises(ValueError,match='qualified156'):m.partial_input()


def test_source_inventory_includes_math_imports_and_actual_card():
    m=driver()
    for path in ('src/neural_shrimp/rigorous_arclength_slices.py','src/neural_shrimp/rigorous_periodic_lyapunov.py',
                 'src/neural_shrimp/models.py','src/neural_shrimp/controls.py',m.CARD):assert path in m.SOURCES
