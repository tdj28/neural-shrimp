import json,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_unit_flip_corner as W
import launch_unit_flip_corner as L

def test_exact_owned_path_and_source_pins():
 assert L.OUT==W.OUT.parent and W.SOURCE=='1a991e0205cf6ddf00adb5b678a14d0f44857a49'and len(W.PINS)==2
 assert all((ROOT/n).is_file()for n in W.FILES)

def test_exact_dyadic_output_and_nonfinite_rejection():
 with ctx.workprec(512):
  a=arb(arb((123,-40)),arb((1,-100)));v=W.pack(a);b=W.A.decode(v);assert b.contains(a)
  with pytest.raises(ValueError):W.pack(arb('nan'))

def test_history_gate_rejects_unproved_root():
 with pytest.raises(ValueError):W.history_gate(dict(passed=False),{}, {},{})


def test_full_synthetic_serialization_fits_declared_forecast():
 with ctx.workprec(512):
  z=W.gradients(*W.fixture(13));C=W.A.eye(55);C[0,54]=arb(1)/3
  data=json.dumps(W.pack(dict(gradient=z,inverse55=W.A.validated_solve(C,W.A.eye(55)))),sort_keys=True,indent=2).encode()
  assert 4*(len(data)+65536)+2*1024**2<16*1024**2
