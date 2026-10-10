import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('n6panel',Path(__file__).parents[1]/'tools/explore_clark_rnn_n6_panel.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_fixed_dimension_cost_panel():
 assert M.SEEDS==[28181,28182,28183,28184] and M.MEASURED==8192 and M.TOTAL==21192
 p=M.projection(dict(batch_seconds=1.,period64_unit_seconds=.001,peak_rss_kib=50000))
 assert p['max_qualifications']==128 and p['conservative_peak_bytes']<512*1024**2 and p['conservative_raw_bytes']<256*1024**2
 assert p['projected_seconds']==pytest.approx(1.8*4*31**2/81+128*8*.001+20)
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0.,-1.])
def test_invalid_profile_closed(bad):
 with pytest.raises(ValueError):M.projection(dict(batch_seconds=bad,period64_unit_seconds=.001,peak_rss_kib=50000))
