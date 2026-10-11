import sys,shutil,subprocess
from fractions import Fraction as F
from pathlib import Path
import pytest
from flint import arb
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_polynomial_complement as D
import launch_native705_polynomial_complement as L


def test_fixed_same_first_quarter_and_original_sources():
    assert D.LEFT==F(-49,2048)and D.RIGHT==F(-17,2048)
    assert F(15,2048)+D.SHIFT-D.HALF==D.LEFT and F(15,2048)+D.SHIFT+D.HALF==D.RIGHT
    r=D.read(ROOT/'docs/reviews/native705-inverse-correlation-diagnosis.json');assert r['outer']==D.DIAG_OUTER and r['passed']and r['native_calls']==0
    assert D.OUT.parent==L.OUT and L.OUT.name=='polynomial001'


def test_exact_acquired_domain_gate_rejects_outward_expansion():
    original=D.ball_json(arb(0,arb(1)/8))
    D.exact_coverage([original],[arb(0,arb(1)/16)],'domain')
    with pytest.raises(ValueError,match='domain'):D.exact_coverage([original],[arb(0,arb(1)/4)],'domain')


def test_closed_source_archive(tmp_path):
    for name in D.FILES:
        q=tmp_path/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,q)
    script="import sys,pathlib;sys.path.insert(0,'tools');import probe_native705_polynomial_complement as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
    subprocess.run([sys.executable,'-I','-B','-c',script],cwd=tmp_path,check=True,capture_output=True)


def test_profile_and_target_phase_caps_and_terminal_domain_contract():
    s=Path(D.__file__).read_text();assert "in (0,36,108,144)"in s and '72<=self.calls<108 or 180<=self.calls<216'in s and 'need(w.calls==216'in s
    assert "weighted_neumann_bound=proof['weighted_neumann_bound']"in s
    t=Path(L.__file__).read_text();assert "'internal_seconds':400"in t and "'external_seconds':415"in t and "'output_cap_bytes':386*1024**2"in t


def test_near_cap_failure_retains_final_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=380*1024**2
    with pytest.raises(ValueError,match='final reserve'):w.save('failed.json',{'a':1})
    w.save('outcome.json',{'passed':False},True);assert D.read(tmp_path/'outcome.json')=={'passed':False}
