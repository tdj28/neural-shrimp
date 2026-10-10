import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/extend_clark_rnn_compact_p5.py'
s=importlib.util.spec_from_file_location('compact_extension',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def fixture(tmp,stop='node_cap',tag='fold:-1:192'):
    (tmp/'run').mkdir();(tmp/'run/result.json').write_text(json.dumps({'traces':[{'branch':'fold','direction':-1,'stop':stop,'nodes':[{'tag':tag}]}]}));p=tmp/'run/000001-node.json';p.write_text(json.dumps({'tag':tag}));return p
def test_only_declared_capped_endpoint(tmp_path):
    p=fixture(tmp_path);assert M.endpoint_paths(tmp_path,'fold',-1,'fold:-1:192')==p
@pytest.mark.parametrize('stop',['domain_edge','reverse_mismatch','qualification_or_arithmetic'])
def test_other_stops_not_reopened(tmp_path,stop):
    fixture(tmp_path,stop)
    with pytest.raises(ValueError):M.endpoint_paths(tmp_path,'fold',-1,'fold:-1:192')
def test_no_arbitrary_other_endpoint(tmp_path):
    fixture(tmp_path,tag='fold:-1:191')
    with pytest.raises(ValueError):M.endpoint_paths(tmp_path,'fold',-1,'fold:-1:192')
