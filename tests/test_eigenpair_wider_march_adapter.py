import importlib.util
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_wider.py';s=importlib.util.spec_from_file_location('wider_march_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_cost_and_complete_inventory():
    m=driver();o={'passed':True,'ticks':127,'status':'one_repaired_wider_chart','elapsed_seconds':21.5,'peak_rss_kib':657180,'stored_bytes_before_final':44089463};r={'passed':True,'ticks':326,'peak_rss_kib':1570596}
    assert m.cost_gate(o,r)['seconds']==364
    for bad in ({'passed':False},{'ticks':126},{'elapsed_seconds':-1},{'peak_rss_kib':float('nan')},{'stored_bytes_before_final':float('inf')}):
        with pytest.raises(ValueError):m.cost_gate(dict(o,**bad),r)
    root=Path(__file__).parents[1];assert set(m.R.SOURCES)<=set(m.SOURCES) and all((root/n).is_file() for n in m.SOURCES)
    assert 'src/neural_shrimp/controls.py' in m.SOURCES
    text=(root/'tools/continue_eigenpair_wider.py').read_text();assert 'result=march(model,parent' in text and 'limit=400' in text and "'fixed1040tick cap'" in text
