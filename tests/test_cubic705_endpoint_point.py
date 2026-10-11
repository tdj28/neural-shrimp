import importlib.util
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('endpoint_point',ROOT/'tools/localize_cubic705_endpoint_point.py');W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)

def test_original_domain_not_outward_decoded_domain():
 with ctx.workprec(480):
  error=[{'mid':[0,0],'rad':[536870913,-40]}];p=[arb(1)];x=[arb(1)+arb((536870913,-40))]
  W.old_domain_contains(error,p,x)
  with pytest.raises(ValueError):W.old_domain_contains(error,p,[x[0]+arb((1,-70))])

def test_source_and_literal_scope():
 assert W.TARGET==W.BASE/'attempt001/run'and W.OLD==W.C.TARGET.parent
 for n in W.SOURCES:assert(ROOT/n).is_file()
 assert 'localize_cubic705_endpoint_point.py'in(ROOT/'tools/launch_cubic705_endpoint_point.py').read_text()
