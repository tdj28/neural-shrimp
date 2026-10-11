import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
p=Path(__file__).resolve().parents[2]/'tools/reviews/audit_norm_cocycle_saved.py'
s=importlib.util.spec_from_file_location('a',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def js(m):return [[dict(a.enc(v),display='test') for v in row] for row in m.tolist()]
def test_pd_uses_saved_basis_but_rejects_negative():
 with ctx.workprec(256):
  a.pd(arb_mat([[2,1],[1,2]]),{'basis':js(a.eye(2))})
  with pytest.raises(ValueError):a.pd(arb_mat([[1,2],[2,1]]),{'basis':js(a.eye(2))})
def test_exact_noncommuting_product():
 with ctx.workprec(256):
  z=arb_mat(2,2);aa=arb_mat([[1,2],[0,1]]);bb=arb_mat([[1,0],[3,1]])
  c,e,g=a.advance([aa,z,z],arb(0),arb(4),[bb,z,z],arb(0),arb(5))
  assert c[0]==aa*bb and c[0]!=bb*aa and e==0 and g==20
def test_remainder_cross_terms():
 with ctx.workprec(256):
  z=arb_mat(1,1);o=arb_mat([[1]])
  c,e,g=a.advance([o,z,z],arb(2),arb(1),[o,z,z],arb(3),arb(1))
  assert e==11
@pytest.mark.parametrize('k',[1,2,3,4])
def test_shared_polynomial_powers(k):
 z=arb_mat(1,1);cs=[z for _ in range(5)];cs[k]=arb_mat([[1]])
 value=a.polynomial(cs)[0,0];assert value.contains(0) and value.contains(1)
 assert (value.contains(-1)) == (k%2==1)
def test_inventory_rejects_unlisted(tmp_path):
 a.put(tmp_path/'manifest.json',{'files':{}});pin=a.sha((tmp_path/'manifest.json').read_bytes());(tmp_path/'extra').write_text('x')
 with pytest.raises(ValueError):a.inventory(tmp_path,pin)
def test_center_error_not_scaled_by_width():
 with ctx.workprec(256):
  j=arb_mat([[arb(1,arb('1/100'))]]);d=arb_mat([[2]])*arb('1/10000')
  err=j-a.mid(j)+(d-a.mid(d))*arb(0,1)
  assert err[0,0].contains(arb('1/100'))
