import importlib.util
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/reanchor_eigenpair_locus.py';s=importlib.util.spec_from_file_location('reanchor_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_cost_gate_and_bad_resource_controls():
    m=driver();p={'passed':True,'status':'four_joined_charts','ticks':508,'elapsed_seconds':78.,'peak_rss_kib':1160000,'stored_bytes_before_final':170000000}
    s={'passed':True,'native_calls':0,'synthetic_dimension':470,'intersections':3,'elapsed_seconds':2.,'peak_rss_kib':200000,'stored_bytes_before_final':400000}
    r=m.cost_gate(p,s);assert r['seconds']<180 and r['raw_bytes']<504*1024**2
    for bad in ({'elapsed_seconds':-1},{'elapsed_seconds':float('nan')},{'ticks':507},{'peak_rss_kib':float('inf')},{'passed':False}):
        with pytest.raises(ValueError):m.cost_gate(dict(p,**bad),s)
    for bad in ({'native_calls':1},{'intersections':2},{'elapsed_seconds':20},{'peak_rss_kib':512*1024},{'stored_bytes_before_final':31*1024**2}):
        with pytest.raises(ValueError):m.cost_gate(p,dict(s,**bad))


def test_full_import_provenance_and_fixed_limits():
    m=driver();root=Path(__file__).parents[1]
    assert set(m.C.ORIGINAL_SOURCES+m.C.EXTRA)<=set(m.SOURCES)
    assert 'src/neural_shrimp/controls.py' in m.SOURCES and 'src/neural_shrimp/models.py' in m.SOURCES
    assert all((root/n).is_file() for n in m.SOURCES)
    s=(root/'tools/reanchor_eigenpair_locus.py').read_text();l=(root/'tools/launch_eigenpair_reanchor.py').read_text()
    assert "'fixed360tick cap'" in s and 'limit=180' in s and 'LIMIT=200' in l
    assert "expected_right='19/32768'" in s and 'compact(full)==previous' in s
    assert "progress['chart_files'].append(name)" in s and 'exist_ok=False' in s
