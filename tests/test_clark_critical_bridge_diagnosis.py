import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import diagnose_clark_critical_bridge as D

def test_diagonal_forcing_and_weighted_proposal():
 a=D.reduce([[.2,0],[0,.3]],[.1,.8],[1,1]);assert a['old_strict_inclusions']==1
 assert np.allclose(a['positive_resolvent'],[.125,.8/.7]);assert a['proposed_exponents']==[-1,2]
 assert a['frozen_old_domain_image_ratio']<1 and a['dominant_coordinates']==[1,0]

def test_exact_dyadic_upper_and_invalid_data():
 assert D.upper_abs(dict(mid=[-3,-2],rad=[1,-3]))==D.F(7,8)
 with pytest.raises(ValueError):D.reduce([[0,-1],[0,0]],[1,1],[1,1])
 with pytest.raises(ValueError):D.reduce([[2,0],[0,2]],[1,1],[1,1])
