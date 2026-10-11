import sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,ctx
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'src')]
import diagnose_native705_fresh_widths as D

def test_literal_closed_domain_no_extra_quarter():
 D.geometry(F(-33,2048),D.WIDTHS,[F(-49,2048),F(-17,2048)])
 with pytest.raises(ValueError):D.geometry(F(-1,2048),D.WIDTHS,[F(-49,2048),F(-17,2048)])
 with pytest.raises(ValueError):D.geometry(F(-33,2048),D.WIDTHS+[F(1,64)],[F(-49,2048),F(-17,2048)])

def test_full18_quotient_includes_actual_derivative_error():
 with ctx.workprec(480):
  small=D.replay([(D.eye(11)/2,D.eye(11)*arb(0,1))for _ in range(18)],F(1,2048));large=D.replay([(D.eye(11)/2,D.eye(11)*arb(0,1))for _ in range(18)],F(1,128))
  assert small['passed'] and large['passed']
  assert D.U.R.original_ball(large['remainder'])>D.U.R.original_ball(small['remainder'])

def test_closing_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=62*1024**2
 with pytest.raises(ValueError):w.save('x.json',{})
 w.save('outcome.json',dict(passed=False),True)


def test_literal_preserved_profile_metadata(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();D.preserved_profile(w)
 assert D.read(tmp_path/'preserved-profile-failure.json')['outcome']['native_calls']==0
