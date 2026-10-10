import importlib.util,json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'tools/explore_response_spectrum.py'
spec=importlib.util.spec_from_file_location('response_driver',P);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

def test_store_finalization_reserve_and_immutability(tmp_path):
    s=d.Store(tmp_path/'a',2*1024**2);s.save('first.json',{'returned':True})
    with pytest.raises(RuntimeError):s.save('too-big.json','x'*1024**2)
    s.save('result.json',{'failure':'cap'},True);s.finish()
    assert not (s.out/'too-big.json').exists()
    assert set(json.loads((s.out/'manifest.json').read_text())['files'])=={'first.json','result.json'}
    assert (s.out/'first.json').stat().st_mode&0o222==0

def profile_fixture(tmp_path,monkeypatch):
    src={'a':{'sha256':'x','bytes':1}};env={'threads':{'x':'1'}};monkeypatch.setattr(d,'sources',lambda:src)
    path=tmp_path/'p';path.mkdir()
    return path,src,env

@pytest.mark.parametrize('field,bad',[('elapsed_seconds',-1),('elapsed_seconds',float('nan')),('max_rss_kib',float('inf')),('projected_panel_seconds',45),('elapsed_seconds',True)])
def test_profile_invalid_resources(tmp_path,monkeypatch,field,bad):
    path,src,env=profile_fixture(tmp_path,monkeypatch)
    (path/'start.json').write_text(json.dumps({'mode':'profile','sources':src,'environment':env}))
    r={'status':'complete','elapsed_seconds':1,'max_rss_kib':100,'projected_panel_seconds':10};r[field]=bad
    (path/'result.json').write_text(json.dumps(r));(path/'manifest.json').write_text(json.dumps({'files':{p.name:d.entry(p)for p in path.iterdir()}}))
    with pytest.raises(ValueError):d.profile_gate(path,d.entry(path/'manifest.json')['sha256'],env)


def test_profile_valid_then_tamper(tmp_path,monkeypatch):
    path,src,env=profile_fixture(tmp_path,monkeypatch)
    (path/'start.json').write_text(json.dumps({'mode':'profile','sources':src,'environment':env}))
    (path/'result.json').write_text(json.dumps({'status':'complete','elapsed_seconds':1,'max_rss_kib':100,'projected_panel_seconds':10}))
    (path/'manifest.json').write_text(json.dumps({'files':{p.name:d.entry(p)for p in path.iterdir()}}));pin=d.entry(path/'manifest.json')['sha256']
    assert d.profile_gate(path,pin,env)
    (path/'result.json').write_text('{}')
    with pytest.raises(ValueError):d.profile_gate(path,pin,env)
