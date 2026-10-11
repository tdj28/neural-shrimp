import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import review_tiny_lm_two_snapshot as A
import numpy as np
import pytest

def test_exact_retained_recurrence_and_corrupt_action():
 initial=np.ones(3);actions=[np.ones(3)*.1 for _ in range(15)];values=[initial]
 for a in actions:values.append(values[-1]-.2*a)
 A.recurrence(np.array(values),initial,.2,actions)
 actions[-1]=actions[-1]*2
 with pytest.raises(ValueError):A.recurrence(np.array(values),initial,.2,actions)

def test_snapshot_gate_requires_all_comparators_and_both_seeds():
 rows=[dict(seed=s,arm=256,direction='gradient',horizon=16,mechanism_qualified=True,alternating_vector_error=.01,latest_vector_error=1.,persistence_vector_error=.5)for s in [28161,28162]]
 assert A.fixed_gate(rows)
 rows[0]['persistence_vector_error']=.05;assert not A.fixed_gate(rows)
 with pytest.raises(ValueError):A.fixed_gate(rows[:1])

def test_nonfinite_acquisition_is_not_a_valid_recurrence():
 with pytest.raises(ValueError):A.recurrence(np.ones((16,3)),np.ones(3),1.,[np.full(3,np.nan)]*15)
