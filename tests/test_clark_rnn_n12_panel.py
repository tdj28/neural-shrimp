import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('n12panel',Path(__file__).parents[1]/'tools/explore_clark_rnn_n12_panel.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_fixed_dimension_cost_panel():
 assert M.SEEDS==[28191,28192,28193,28194] and M.MEASURED==8192 and M.TOTAL==21192
 p=M.projection(dict(batch_seconds=1.,period64_unit_seconds=.001,peak_rss_kib=50000))
 assert p['max_qualifications']==128 and p['conservative_peak_bytes']<512*1024**2 and p['conservative_raw_bytes']<512*1024**2
 assert p['projected_seconds']==pytest.approx(1.8*4*31**2/81+128*8*.001+20)
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0.,-1.])
def test_invalid_profile_closed(bad):
 with pytest.raises(ValueError):M.projection(dict(batch_seconds=bad,period64_unit_seconds=.001,peak_rss_kib=50000))


def test_fixed_dimension_seed_binding():
 assert M.N==12 and M.PROFILE_SEED==28189
 source=Path(M.__file__).read_text()
 assert 'inputs(N,seed)' in source and "f'n{N}-s{seed}'" in source and 'np.zeros(N)' in source
 assert 'inputs(6,' not in source and 'np.zeros(6)' not in source

def test_n12_storage_explicitly_larger_than_n6():
 p=M.projection(dict(batch_seconds=1.,period64_unit_seconds=.001,peak_rss_kib=50000))
 assert 256*1024**2<p['conservative_raw_bytes']<512*1024**2

@pytest.mark.parametrize('field,value',[('batch_seconds',100.),('period64_unit_seconds',100.),('peak_rss_kib',500000)])
def test_resource_forecast_rejects(field,value):
 values=dict(batch_seconds=1.,period64_unit_seconds=.001,peak_rss_kib=50000);values[field]=value
 with pytest.raises(ValueError):M.projection(values)
