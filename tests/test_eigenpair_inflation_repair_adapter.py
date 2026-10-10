import importlib.util
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).parents[1]/'tools/repair_eigenpair_wider.py';s=importlib.util.spec_from_file_location('repair_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_cost_and_inventory():
    m=driver();old={'passed':False,'ticks':127,'status':'wider_proof_gate_unresolved','elapsed_seconds':20,'peak_rss_kib':656552,'stored_bytes_before_final':43240028}
    assert m.cost_gate(old)['seconds']==60
    for bad in ({'passed':True},{'ticks':126},{'elapsed_seconds':-1},{'peak_rss_kib':float('nan')},{'stored_bytes_before_final':float('inf')}):
        with pytest.raises(ValueError):m.cost_gate(dict(old,**bad))
    root=Path(__file__).parents[1];assert set(m.I.SOURCES)<=set(m.SOURCES) and all((root/n).is_file() for n in m.SOURCES)
    assert 'src/neural_shrimp/controls.py' in m.SOURCES
    text=(root/'tools/repair_eigenpair_wider.py').read_text();assert 'chart=repair(model,parent,proposal' in text and 'limit=90' in text and "'fixed130tick cap'" in text
