import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
from flint import arb,arb_mat,ctx
import pytest
import review_endpoint_attraction as R

def test_positive_congruence():assert R.positive(arb_mat([[2,1],[1,2]]),arb_mat([[1,0],[0,1]]))['passed']
def test_semidefinite_fails_strict_gate():assert not R.positive(arb_mat([[1,1],[1,1]]),arb_mat([[1,0],[0,1]]))['passed']
def test_indefinite_fails():assert not R.positive(arb_mat([[1,2],[2,1]]),arb_mat([[1,0],[0,1]]))['passed']
def test_unstable_cycle_fails_direct_decrease():
 P=arb_mat([[1]]);J=arb_mat([[arb(11)/10]]);assert not R.positive(P-J.transpose()*P*J,P)['passed']
def test_singular_congruence_not_accepted():
 with pytest.raises((ZeroDivisionError,ValueError)):R.positive(arb_mat([[1]]),arb_mat([[0]]))
