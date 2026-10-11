from fractions import Fraction as F
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import clark_subset_radius_math as H

def test_successful_nonuniform_subset_despite_old_row_norm_above_one():
 A=[[F(0),F(3,2)],[F(0),F(0)]];b=[F(1,10),F(1,10)];a=[F(1,8),F(1,8)];result=H.propose(A,b,a);v=result['verification'];assert v['passed']and v['contraction']<1 and all(0<z<=1 for z in v['proposal'])

def test_floating_proposal_never_overrides_exact_domain_or_join():
 A=[[F(0)]];b=[F(1,10)];a=[F(1,2)]
 assert not H.verify(A,b,a,[F(10000000000000001,10000000000000000)])['passed']
 assert not H.verify(A,b,a,[F(1,2)])['passed']
 assert H.verify(A,b,a,[F(3,4)])['passed']

def test_strict_root_and_contraction_and_infeasible_label():
 assert not H.verify([[F(1)]],[F(0)],[F(0)],[F(1)])['passed']
 r=H.propose([[F(2)]],[F(1,10)],[F(1,8)]);assert not r['proposal']['success']and r['verification']is None

def test_full51_unrelated_proposal():
 n=51;A=[[F(int(i==j),8)for j in range(n)]for i in range(n)];r=H.propose(A,[F(1,16)]*n,[F(1,4)]*n);assert r['verification']['passed']


def test_floating_result_retained_before_exact_decision(monkeypatch):
 seen=[]
 def fail(*args):raise RuntimeError('strict verification stopped')
 monkeypatch.setattr(H,'verify',fail)
 import pytest
 with pytest.raises(RuntimeError,match='strict verification'):
  H.propose([[F(0)]],[F(1,10)],[F(1,4)],observe=seen.append)
 assert len(seen)==1 and seen[0]['success'] and len(seen[0]['x'])==2
