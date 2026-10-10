import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import audit_chart156_stability_saved as M
import pytest
from fractions import Fraction

def rows():return [[str(Fraction(k,1024)),str(Fraction(k+1,1024))]for k in range(352,360)]
def test_exact_coverage():assert M.coverage(rows())==['11/32','45/128']
def test_no_missing_slice():
 with pytest.raises(ValueError):M.coverage(rows()[:-1])
def test_no_gap_or_duplicate():
 r=rows();r[3]=r[4]
 with pytest.raises(ValueError):M.coverage(r)
def test_schema_requires_all_stages(tmp_path):
 with pytest.raises(ValueError,match='161'):M.grouped(tmp_path,0)


def test_reused_input_authentication_precedes_read(monkeypatch):
 events=[];inventories=[]
 def auth(*args):
  assert args==M.R.PINS[2];events.append('authenticate');return {'verified':True},None,None
 def read(path):
  assert events==['authenticate'];events.append('read');return {'inputs':True}
 monkeypatch.setattr(M,'authenticate',auth);monkeypatch.setattr(M,'read',read)
 assert M.authenticated_reused_inputs(inventories)=={'inputs':True}
 assert inventories==[{'verified':True}] and events==['authenticate','read']

def test_failed_reused_authentication_never_reads(monkeypatch):
 def auth(*args):raise ValueError('manifest mismatch')
 def forbidden(path):raise AssertionError('read before authentication')
 monkeypatch.setattr(M,'authenticate',auth);monkeypatch.setattr(M,'read',forbidden)
 with pytest.raises(ValueError,match='manifest mismatch'):M.authenticated_reused_inputs([])
