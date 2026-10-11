import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat
from neural_shrimp.rigorous_orbits import vector_json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
import review_known_endpoint_math as A

def test_neumann_contracts_same_known_root():
    old=vector_json([arb(0,1),arb(0,1)]);whole=vector_json([arb(0,2),arb(0,2)]);v=A.localize(arb_mat([[1,0],[0,1]]),arb_mat([[1,0],[0,1]]),arb_mat([[0],[0]]),[arb(0),arb(0)],old,whole,[arb(1),arb(1)])
    assert v['passed']and all(A.M.interval(x)==(0,0)for x in v['box'])

def test_outside_domain_and_noncontractive_matrix_rejected():
    old=vector_json([arb(0,1)]);whole=vector_json([arb(0,2)])
    with pytest.raises(ValueError):A.localize(arb_mat([[1]]),arb_mat([[0]]),arb_mat([[0]]),[arb(0)],old,whole,[arb(1)])
    with pytest.raises(ValueError):A.localize(arb_mat([[1]]),arb_mat([[1]]),arb_mat([[0]]),[arb(3)],old,whole,[arb(1)])
