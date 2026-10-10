import importlib.util,json
from pathlib import Path
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/extend_clark_rnn_compact_p12.py';s=importlib.util.spec_from_file_location('p12extend',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_unresolved_fold_cannot_enter_pal(tmp_path):
 m=module();(tmp_path/'run').mkdir();(tmp_path/'run/result.json').write_text(json.dumps({'fold_associations':[{'tag':'root','passed':False}]}))
 with pytest.raises(ValueError,match='association'):m.associated_fold_path(tmp_path,tmp_path,'root')

def test_only_fixed_cap_endpoint_allowed(tmp_path):
 m=module();(tmp_path/'run').mkdir();(tmp_path/'run/result.json').write_text(json.dumps({'traces':[{'branch':'flip','direction':1,'stop':'forward_guard','nodes':[{'tag':'flip:1:192'}]}]}))
 with pytest.raises(ValueError,match='capped endpoint'):m.endpoint_paths(tmp_path,'flip',1,'flip:1:192')

def test_fixed_seeds_and_same_domain():
 m=module();assert m.SEEDS==[('fold',-1,'g16-dir-1-fold',1),('fold',1,'g16-dir-1-fold',1),('flip',1,'flip:1:192',-1)];assert m.in_domain([0,0,0,0,3.3/5,.77/.1,0,0,0,0])
