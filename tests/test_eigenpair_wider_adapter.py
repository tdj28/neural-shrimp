import importlib.util
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/widen_eigenpair_locus.py';s=importlib.util.spec_from_file_location('wider_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_cost_and_prospective_failclosed_support_gate():
    m=driver();a={'passed':True,'status':'four_joined_charts','ticks':508,'elapsed_seconds':76,'peak_rss_kib':1137520,'stored_bytes_before_final':170000000};d={'passed':True,'ticks':0,'summary':{'double_width_proxy_below_one':True}};r={'passed':True,'ticks':326,'peak_rss_kib':1570596}
    assert m.cost_gate(a,d,r)['seconds']==96
    for bad in ({'ticks':507},{'elapsed_seconds':-1},{'peak_rss_kib':float('nan')},{'stored_bytes_before_final':float('inf')}):
        with pytest.raises(ValueError):m.cost_gate(dict(a,**bad),d,r)
    with pytest.raises(ValueError):m.cost_gate(a,dict(d,summary={'double_width_proxy_below_one':False}),r)


def test_source_inventory_and_no_old_math_mutation():
    m=driver();root=Path(__file__).parents[1]
    assert set(m.D.SOURCES)<=set(m.SOURCES) and all((root/n).is_file() for n in m.SOURCES)
    assert 'src/neural_shrimp/controls.py' in m.SOURCES
    s=(root/'tools/widen_eigenpair_locus.py').read_text();assert 'limit=150' in s and "'fixed260tick cap'" in s
    assert "result=widen(model,parent" in s and 'profile' not in m.EXTRA
