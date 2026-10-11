import json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_tiny_lm_horizon as W


def test_complete_profile_counts_and_forecast():
    rows=[dict(arm=a,times=dict(hvp=[.1,.2],ce=[.01,.02]))for a in(8,256)]
    assert W.forecast(rows)==pytest.approx(20+4*(36*.2+16*.02))
    rows[1]['arm']=8
    with pytest.raises(ValueError):W.forecast(rows)


def test_native_attempt_inputs_survive_failure(tmp_path):
    w=W.Work(tmp_path/'run','profile')
    class Broken:
        def hvp(self,*args):raise RuntimeError('synthetic native interruption')
    with pytest.raises(RuntimeError):w.call('hvp',Broken(),np.ones(3),np.zeros(3))
    assert w.attempted['hvp']==1 and w.returned['hvp']==0
    assert np.array_equal(np.load(w.out/'operations/hvp-001-theta.npy'),np.ones(3))
    assert np.array_equal(np.load(w.out/'operations/hvp-001-vector.npy'),np.zeros(3))
    assert json.loads((w.out/'operations/hvp-001-attempt.json').read_text())['returned']is False


def test_closing_reserve_and_fixed_array_storage_budget(tmp_path):
    w=W.Work(tmp_path/'run','target');w.used=124*1024**2-8
    with pytest.raises(ValueError):w.json('normal.json',dict(message='x'*30))
    w.json('result.json',dict(passed=False),True);w.json('manifest.json',dict(files={}),True)
    vectors=72*3+32+32*(6+2+5+2*5)
    upper=vectors*(15776*8+128)+4*1024**2
    assert upper<124*1024**2


def test_evaluation_requires_closed_all_predictors(tmp_path):
    w=W.Work(tmp_path/'run','target')
    with pytest.raises(ValueError,match='sealed'):W.evaluate_predictions(w,[])


def test_isolated_full_archive_module_closure(tmp_path):
    for n in W.FILES:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    code="""import sys
from pathlib import Path
import explore_tiny_lm_horizon as W
root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.FILES),loaded-set(W.FILES)
assert Path(W.__file__).resolve().is_relative_to(root)
"""
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)


def test_only_exact_historical_container_exemption(tmp_path,monkeypatch):
    root=tmp_path/'legacy';root.mkdir();data=root/'data.json';data.write_text('{}');data.chmod(0o444)
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(files={'data.json':W.B.entry(data)})));manifest.chmod(0o444)
    pin=W.B.sha(manifest);monkeypatch.setattr(W.B,'PARENTS',dict(profile=(root,pin)))
    assert W.immutable(root,pin)['historical_writable_container']
    monkeypatch.setattr(W.B,'PARENTS',{})
    with pytest.raises(ValueError):W.immutable(root,pin)
    root.chmod(0o555);assert not W.immutable(root,pin)['historical_writable_container']
    data.chmod(0o644)
    with pytest.raises(ValueError):W.immutable(root,pin)
