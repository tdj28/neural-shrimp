import numpy as np
import pytest
from neural_shrimp.unit_pair_conditioning import middle_radius,positive_scaling

def test_wide_row_sum_can_have_small_weighted_radius():
 b=np.array([[.1,100],[.0001,.1]])
 r=positive_scaling(b);assert .199<r['collatz_min']<=r['collatz_max']<.201
 w=np.array(r['weights']);assert np.max((b@w)/w)<.201

def test_unstable_radius_stays_unstable():
 r=positive_scaling([[2,.1],[.1,2]]);assert r['collatz_min']>1

def test_zero_matrix_and_negative_or_nonfinite_reject():
 assert positive_scaling([[0,0],[0,0]])['collatz_max']==0
 for b in ([[0,-1],[1,0]],[[np.nan,0],[0,0]]):
  with pytest.raises(ValueError):positive_scaling(b)

def test_native_dyadic_decode_and_invalid_ball():
 assert middle_radius({'mid':[3,-2],'rad':[1,-5]})==(.75,.03125)
 with pytest.raises(ValueError):middle_radius({'mid':[3,-2],'rad':[-1,-5]})
