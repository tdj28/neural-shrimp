import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];p=ROOT/'tools/reviews/review_unit_flip_corner.py'
s=importlib.util.spec_from_file_location('corner_audit_driver',p);D=importlib.util.module_from_spec(s);s.loader.exec_module(D)

def test_full_same_root_uses_exact_serialized_endpoints():
    def box(rad):return [dict(mid=[0,0],rad=[1,rad])for _ in range(55)]
    old=dict(box=box(-80));fresh=dict(box=box(-84));identity=dict(original_box=old['box'],fresh_box=fresh['box'],fresh_inside_original=[True]*55)
    D.strict_same_root(old,fresh,identity);fresh['box'][17]=dict(mid=[1,-80],rad=[1,-84]);identity['fresh_box']=fresh['box']
    with pytest.raises(ValueError):D.strict_same_root(old,fresh,identity)

def test_isolated_declared_source_import():
    with tempfile.TemporaryDirectory()as name:
        root=Path(name)
        for n in D.FILES:
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
        code="import importlib.util; p='tools/reviews/review_unit_flip_corner.py'; s=importlib.util.spec_from_file_location('audit',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.main)"
        subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True,capture_output=True)

def test_literal_inventory_metadata_and_owned_caps():
    card=D.read(D.CARD)
    for spec in card.values():
        root=Path(spec['root']);assert D.entry(root/'manifest.json')['sha256']==spec['outer']and D.entry(root/'run/manifest.json')['sha256']==spec['worker']
        for n,v in spec['files'].items():assert D.entry(root/n)==v
    source=(ROOT/'tools/launch_unit_flip_corner_review.py').read_text();assert '18*1024**2'in source and '2*1024**3'in source and 'tools/reviews/review_unit_flip_corner.py'in source

def test_closing_reserve_preserves_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=14*1024**2-8
    with pytest.raises(ValueError):w.save('large.json',dict(a='x'*30))
    w.save('outcome.json',dict(passed=False),True);assert not D.read(tmp_path/'outcome.json')['passed']
