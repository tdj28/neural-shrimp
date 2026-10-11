import copy,hashlib,io,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_tiny_lm_common_subspace as W
import launch_tiny_lm_common_subspace as L


def test_null_authority_and_literal_gate_mutation_block():
    c=W.read(ROOT/W.CARD)
    with pytest.raises(ValueError):W.card_check(c)
    c['status']='reviewed';c['independent']=dict(root='/tmp/control',source='a'*40,outer='b'*64,worker='c'*64);W.card_check(c)
    bad=copy.deepcopy(c);bad['gates']['fidelity']=.1
    with pytest.raises(ValueError):W.card_check(bad)
    assert W.OUT.parent==L.OUT


def test_complete_synthetic_and_retained_intermediates():
    saved={};out=W.synthetic(64,lambda k,v:saved.__setitem__(k,v.copy()))
    assert out['passed'] and out['rank']==16 and len(out['fidelity'])==3
    assert saved['basis'].shape==(64,16) and saved['matched_eigenvalues'].shape==(16,16)


def test_exact_object_pointer_and_alias(tmp_path):
    a=np.arange(4,dtype=np.float64);buf=io.BytesIO();np.save(buf,a,allow_pickle=False);raw=buf.getvalue();sha=hashlib.sha256(raw).hexdigest();name='objects/'+sha+'.npy';(tmp_path/'objects').mkdir();(tmp_path/name).write_bytes(raw)
    ptr=dict(path=name,sha256=sha,bytes=len(raw),shape=[4],dtype='float64');files={name:dict(bytes=len(raw),sha256=sha)}
    assert np.array_equal(W.load_object(tmp_path,ptr,files),a)
    bad=dict(ptr,shape=[3])
    with pytest.raises(ValueError):W.load_object(tmp_path,bad,files)
    bad=dict(ptr,path='../escape.npy')
    with pytest.raises(ValueError):W.load_object(tmp_path,bad,files)


def test_closing_reserve_survives_time_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=60*1024**2
    with pytest.raises(ValueError):w.save('bad.json',{})
    w.start-=61
    w.save('result.json',dict(passed=False,error='retained'),True)
    assert W.read(tmp_path/'result.json')['passed'] is False


def test_isolated_archive_and_preimport_worker_cap(tmp_path):
    for name in W.FILES:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_tiny_lm_common_subspace as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES);assert 'torch' not in sys.modules"
    p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
    # Exercise the real script under the inherited512MiB soft cap. It must
    # finish scientific imports, then reject the dirty/prepared/null card.
    code="import resource,runpy,os;resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,3*1024**3));os.nice(10);runpy.run_path("+repr(str(ROOT/'tools/explore_tiny_lm_common_subspace.py'))+",run_name='__main__')"
    p=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True)
    assert p.returncode!=0 and any(x in p.stderr for x in ['clean exact source','literal reviewed fixed design','sealed independent temporal-mean authority required']),p.stderr
