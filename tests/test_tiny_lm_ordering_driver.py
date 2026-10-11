import json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_tiny_lm_ordering as W


def test_exact_native_caps_and_no_loss_or_gradient(tmp_path):
    p=W.Work(tmp_path/'p','profile');t=W.Work(tmp_path/'t','target')
    assert p.cap==dict(hvp=4)and t.cap==dict(hvp=160)
    rows=[dict(arm=a,times=[.1,.2])for a in(8,256)]
    assert W.profile_forecast(rows)==pytest.approx(84.)


def test_failed_native_preserves_attempt_inputs(tmp_path):
    w=W.Work(tmp_path/'run','target')
    class Broken:
        def hvp(self,*args):raise RuntimeError('synthetic HVP failure')
    with pytest.raises(RuntimeError):w.call('hvp',Broken(),np.ones(3),np.zeros(3))
    assert w.attempted==dict(hvp=1)and w.returned==dict(hvp=0)
    assert np.array_equal(np.load(w.out/'operations/hvp-001-theta.npy'),np.ones(3))
    assert json.loads((w.out/'operations/hvp-001-attempt.json').read_text())['returned']is False


def test_full_target_retained_array_budget_and_closing_reserve(tmp_path):
    #160 triples + traces(5+17)*2*4 + 16 odd/even/CE-gradient triples.
    vectors=160*3+(5+17)*2*4+16*3
    assert vectors*(15776*8+128)+8*1024**2<124*1024**2
    w=W.Work(tmp_path/'run','target');w.used=124*1024**2-8
    with pytest.raises(ValueError):w.json('normal.json',dict(message='x'*40))
    w.json('result.json',dict(passed=False),True)


def test_isolated_complete_archive_closure(tmp_path):
    for n in W.FILES:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    code="""import sys
from pathlib import Path
import explore_tiny_lm_ordering as W
r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)}
assert loaded<=set(W.FILES),loaded-set(W.FILES)
assert Path(W.__file__).resolve().is_relative_to(r)
"""
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
