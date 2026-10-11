from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.known_endpoint_localization import localize
from neural_shrimp.rigorous_eigenpair_march import decode

def test_known_nonlinear_root_localized_without_reclaiming_existence():
 with ctx.workprec(256):
  r=localize(arb_mat([[arb('1/2')]]),arb_mat([[arb(2,arb('1/8'))]]),arb_mat([[0]]),[arb(1)],[arb(1,arb('1/100'))],[arb(1,arb('1/10'))],[arb(1)])
  assert decode(r['box'][0]).contains(1) and decode(r['box'][0]).rad()<arb('1e-60')

def test_known_root_must_share_whole_derivative_domain():
 with pytest.raises(ValueError,match='mean-value'):localize(arb_mat([[1]]),arb_mat([[1]]),arb_mat([[0]]),[arb(0)],[arb(1)],[arb(0,arb('1/2'))],[arb(1)])

def test_nonzero_rhs_remains_and_singular_derivative_rejected():
 r=localize(arb_mat([[1]]),arb_mat([[1]]),arb_mat([[arb('-1/8')]]),[arb(0)],[arb('1/8',arb('1/32'))],[arb(0,arb(1))],[arb(1)])
 assert decode(r['box'][0]).contains(arb('1/8')) and not decode(r['box'][0]).contains(0)
 with pytest.raises(ValueError,match='Neumann'):localize(arb_mat([[1]]),arb_mat([[0]]),arb_mat([[0]]),[arb(0)],[arb(0)],[arb(0,arb(1))],[arb(1)])

def test_fixed_sources_dimensions_and_native_counts():
 from pathlib import Path
 import sys
 root=Path(__file__).parents[1];sys.path.insert(0,str(root/'tools'));import localize_native_unit_pair_endpoint as E;import launch_native_unit_pair_endpoint as L
 assert E.TARGET==L.C.TARGET and E.PARENT.name=='attempt003-target'
 assert 'src/neural_shrimp/rigorous_parameter_localization.py'in E.SOURCES
 assert 'ticks[0]==144'in (root/'tools/localize_native_unit_pair_endpoint.py').read_text()
