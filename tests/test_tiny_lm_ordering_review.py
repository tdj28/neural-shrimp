import sys,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_tiny_lm_ordering as A

def test_independent_exact_reversal_and_cancellation():
    j0=np.array([[.8,.1],[.1,.4]]);j1=np.array([[.7,-.2],[-.2,.6]]);p=j1@j0;r=j0@j1;u=np.array([1.,0.]);g=np.array([.2,-.3]);f=j0@j0@u
    out,a,b=A.reduce(u,p@u,r@u,f,g)
    assert out['algebra_qualified']and np.allclose(a+b,p@u-f)
    assert out['squared_error']==pytest.approx(out['squared_odd']+out['squared_even']+out['twice_cross_term'])
    assert A.reciprocity(np.eye(2),p.T,r.T)['passed']
    bad=A.reduce(u,p@u,r@u+u,f,g)[0];assert not bad['algebra_qualified']
    assert not A.reciprocity(np.eye(2),p.T,r.T+.1)['passed']

def test_zero_error_and_material_ratios_are_not_percentages():
    u=np.array([1.,0.]);p=np.array([1.,1.]);r=np.array([1.,-1.]);out,a,b=A.reduce(u,p,r,p,u)
    assert out['algebra_qualified']and out['frozen_error_norm']==0 and out['odd_to_error']>1
    assert out['odd_even_inner_product']<0

def test_fsum_sensitive_summation_and_strict_flags():
    assert A.dot(np.array([1e16,1.,-1e16]),np.ones(3))==1
    with pytest.raises(ValueError):A.compare(dict(passed=True),dict(passed=False))
    with pytest.raises(ValueError):A.compare(dict(odd_to_error=.5),dict(odd_to_error=.4))

def test_complete_zero_native_archive_import(tmp_path):
    for n in A.FILES:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    code="import sys,pathlib;sys.path.insert(0,'tools');import review_tiny_lm_ordering as a;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(a.FILES);assert 'torch'not in sys.modules;assert not any(n.startswith('neural_shrimp')for n in sys.modules)"
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_source_pins_and_owned_bounds_agree():
    import launch_tiny_lm_ordering_review as L
    assert L.OUT/A.OUT.name==A.OUT
    text=(ROOT/'tools/launch_tiny_lm_ordering_review.py').read_text()
    assert "'internal_seconds':60"in text and "'external_seconds':75"in text and '>75'in text
    assert len(A.PINS)==6 and all(len(v)==64 for v in A.PINS.values())


def test_distinct_exact_executions_and_historical_schema(tmp_path,monkeypatch):
    assert A.EXEC!=A.PROFILE_EXEC
    files={'a.py':dict(bytes=1,sha256='a'*64)}
    rows=[dict(path=k,**v)for k,v in files.items()]
    assert A.inventory_rows(rows)==files==A.inventory_rows(files)
    with pytest.raises(ValueError):A.inventory_rows(rows+rows)
    with pytest.raises(ValueError):A.inventory_rows([dict(path='a.py',bytes=1)])
    monkeypatch.setattr(A,'read',lambda p:dict(source_commit=A.PROFILE_EXEC,sources={}))
    assert A.source(tmp_path,A.PROFILE_EXEC)['source_commit']==A.PROFILE_EXEC
    with pytest.raises(ValueError):A.source(tmp_path,A.EXEC)
    with pytest.raises(ValueError):A.source(tmp_path,'0'*40)
