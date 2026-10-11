import os,subprocess,sys
from pathlib import Path
from flint import arb,ctx
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_cubic_polynomial_domain
sys.path.insert(0,str(Path(__file__).parent))
import review_cubic_center
sys.path.insert(0,str(Path(__file__).parent))
import diagnose_cubic_polynomial_domains as D


def test_screen_retains_all_six_even_on_first_failure(monkeypatch):
    with ctx.workprec(480):
        z=D.B.vector_json([arb(0)]*705);one=D.B.vector_json([arb(1)]*705)
        data=dict(domain=dict(center=z,slope=one,curvature=z,cubic=z,parameter='0'))
        domains=[dict(domain=dict(index=i,endpoints=['0','1/128']))for i in range(1,7)]
        def read(path):
            i=int(path.parent.name);return dict(domain=domains[i-1]['domain'],original_offset=['0','1/128'],predictor_tangent=z if i==1 else one,segment=D.B.vector_json([arb(0,1)]*705))
        monkeypatch.setattr(D,'read',read)
        class W:
            def __init__(self):self.saved={}
            def save(self,n,v):self.saved[n]=v
            def tick(self):pass
        w=W();r=D.screen(w,Path('/not-read'),domains,data)
        assert not r['passed']and len(r['domains'])==len(w.saved)==6
        assert r['domains'][0]['nested_failures']==705 and all(v['passed']for v in r['domains'][1:])


def test_screen_resource_reserve_and_source_closure(tmp_path,monkeypatch):
    monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=60*1024**2
    with pytest.raises(ValueError):w.save('ordinary.json',{})
    w.save('outcome.json',dict(passed=False),True)
    assert D.read(tmp_path/'outcome.json')['passed']is False


def test_isolated_source_closure_and_explicit_supervisor_caps(tmp_path):
    here=Path(__file__).parent
    for name in D.FILES:
        overlay=here/Path(name).name;source=overlay if overlay.exists()else ROOT/name
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import diagnose_cubic_polynomial_domains as d;import launch_cubic_polynomial_domains as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
    p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
    text=(ROOT/'tools/reviews/launch_cubic_polynomial_domains.py').read_text()
    assert "'internal_seconds':120"in text and "'external_seconds':135"in text
    assert "'worker_rss_kib':2*1024**2"in text and "'output_cap_bytes':66*1024**2"in text
