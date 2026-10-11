import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'tools/qualify_eigenpair_shooting_fold.py'
spec=importlib.util.spec_from_file_location('d',P);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,True,'1'])
def test_resource_failclosed(bad):assert not d.finite_bound(bad,10)

def test_resource_finite():assert d.finite_bound(0,10) and not d.finite_bound(10,10)

@pytest.mark.parametrize('name',['/tmp/a','../a','x/../../a'])
def test_path_escape(tmp_path,name):
 with pytest.raises(ValueError):d.path(tmp_path,name)

def test_profile_dimensions():
 j,x=d.synthetic();assert j.nrows()==470 and len(x)==470
 assert all(x[26*k+13]==1 for k in range(18))
