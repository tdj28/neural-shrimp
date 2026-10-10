import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('n12p6',Path(__file__).parents[1]/'tools/refine_clark_rnn_n12_p6.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_fixed_selection_both_starts_cap_and_no_other_period():
    p=np.repeat(np.array([6]*100+[12]*50+[24]*50+[3]*20),2)
    selected=M.selection(p)
    assert len(selected)==96 and set(p[selected])=={6,12,24}
    assert all(selected[i+1]==selected[i]+1 for i in range(0,len(selected),2))
def test_disagreement_excluded():
    assert M.selection(np.array([6,12,6,6])).tolist()==[2,3]
def test_scaling_has_horizon_and_batch():
    r=M.projection({'batch_seconds':1.,'period64_unit_seconds':.001,'peak_rss_kib':50000})
    assert r['projected_seconds']==pytest.approx(2*(2601/81)+96*8*.001+20)
    assert r['conservative_raw_bytes']<512*1024**2


def test_exact_dimension_seed_crop_bindings():
 assert M.N==12 and M.SEED==28192
 source=Path(M.__file__).read_text()
 assert 'inputs(N,SEED)' in source and "f'run/n{N}-s{SEED}'" in source and "'grid_indices':[11,25,18,30]" in source
 assert 'inputs(6,' not in source

@pytest.mark.parametrize('key,bad',[('batch_seconds',float('nan')),('batch_seconds',100.),('period64_unit_seconds',100.),('peak_rss_kib',2000000)])
def test_resource_admission_fails_closed(key,bad):
 r={'batch_seconds':1.,'period64_unit_seconds':.001,'peak_rss_kib':50000};r[key]=bad
 with pytest.raises(ValueError):M.projection(r)
