import importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import diagnose_cubic_full_domains as D


class Work:
    def __init__(self):self.saved={}
    def tick(self):pass
    def save(self,name,row):self.saved[name]=row


def test_all_six_domain_rejections_are_retained_without_bypass(monkeypatch):
    seen=[]
    def shared(w,run,spec,data):
        i=spec['domain']['index'];seen.append(i)
        w.save(f'domains/{i:03d}/pre-tangent-comparisons.json',{'passed':i!=2})
        if i==2:raise ValueError('actual segment uncovered')
        return dict(whole=[0]*705,tangent=[0]*705,derivatives=[None]*18)
    monkeypatch.setattr(D.C,'shared_domain',shared)
    w=Work();result=D.screen(w,Path('/never'),[dict(domain=d,start=0)for d in D.B.domains()],{})
    assert seen==list(range(1,7))and not result['passed']and len(result['domains'])==6
    assert result['domains'][1]['error']=='actual segment uncovered'
    assert all(f'domains/{i:03d}/screen-result.json'in w.saved for i in range(1,7))


def test_resource_failure_never_becomes_a_domain_rejection(monkeypatch):
    seen=[]
    def shared(*args):seen.append(1);raise TimeoutError('owned cap')
    monkeypatch.setattr(D.C,'shared_domain',shared)
    with pytest.raises(TimeoutError,match='owned cap'):D.screen(Work(),Path('/never'),[dict(domain=d,start=0)for d in D.B.domains()],{})
    assert len(seen)==1


def test_reviewed_literal_card_and_reserved_finalization(tmp_path,monkeypatch):
    card=dict(status='reviewed',failed_replay=D.FAILURE,domains=list(range(1,7)),precision=480,native_calls=0,worker_seconds=360,worker_rss_gib=4,worker_raw_mib=256)
    D.card_check(card)
    for key,value in [('status','prepared'),('native_calls',1),('domains',[1,2]),('failed_replay',dict(D.FAILURE,source='0'*40))]:
        with pytest.raises(ValueError):D.card_check(dict(card,**{key:value}))
    monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=252*1024**2-1
    with pytest.raises(OSError):w.save('too-close.json',{'a':1})
    w.save('outcome.json',{'passed':False},True)
    assert json.loads((tmp_path/'outcome.json').read_text())=={'passed':False}


def test_isolated_source_archive_import_closure(tmp_path):
    for name in D.FILES:
        src=ROOT/name;assert src.is_file(),name
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(src.read_bytes())
    script="""import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'tools/reviews'),str(root/'tools'),str(root/'src')]
import diagnose_cubic_full_domains as D
loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(D.FILES),sorted(loaded-set(D.FILES))
print('archive closed')
"""
    p=subprocess.run([sys.executable,'-I','-c',script,str(tmp_path)],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert p.returncode==0,p.stderr
