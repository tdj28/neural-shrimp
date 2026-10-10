import numpy as np
import pytest
from neural_shrimp.utility_paired_geometry import paired
@pytest.mark.parametrize('rates',[[1.,1.],[100.,1.],[1.,100.],[.01,.02]])
def test_both_norms_and_cap(rates):
 r=paired(np.arange(1.,14.),np.array(rates));assert r['qualified'];assert min(r['block_fractions'].ravel())>=1/16-1e-14;assert max(r['block_fractions'].ravel())<=3/16+1e-14
 assert np.linalg.norm(r['update_plus'])==pytest.approx(np.linalg.norm(r['update_minus']))
 assert np.linalg.norm(r['delta_plus'])==pytest.approx(np.linalg.norm(r['delta_minus']))
def test_equal_rates_remove_first_order_confound():assert abs(paired(np.arange(1.,14.),np.ones(2))['first_order_loss_difference_plus_minus'])<1e-12
def test_unequal_rates_retain_first_order_confound():assert abs(paired(np.arange(1.,14.),np.array([10.,1.]))['first_order_loss_difference_plus_minus'])>.1
@pytest.mark.parametrize('g',[np.zeros(13),np.r_[np.ones(9),np.zeros(4)],np.full(13,np.nan)])
def test_bad_blocks_abstain(g):
 with pytest.raises(ValueError):paired(g,np.ones(2))
def test_equal_update_uniform_does_not_match_pulse():
 r=paired(np.ones(13),np.ones(2));assert r['uniform_equal_update_pulse_norm']<r['paired_pulse_norms'][0]
