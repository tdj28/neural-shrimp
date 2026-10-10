import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('compactp14',Path(__file__).parents[1]/'tools/refine_clark_rnn_compact_p14.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_fixed_selection_both_starts_cap_and_no_other_period():
    p=np.repeat(np.array([14]*100+[28]*50+[56]*50+[7]*20),2)
    selected=M.selection(p)
    assert len(selected)==96 and set(p[selected])=={14,28,56}
    assert all(selected[i+1]==selected[i]+1 for i in range(0,len(selected),2))
def test_disagreement_excluded():
    assert M.selection(np.array([14,28,14,14])).tolist()==[2,3]
def test_scaling_has_horizon_and_batch():
    r=M.projection({'batch_seconds':1.,'period64_unit_seconds':.001,'peak_rss_kib':50000})
    assert r['projected_seconds']==pytest.approx(2*(2601/81)*(34192/21192)+96*8*.001+20)
    assert r['conservative_raw_bytes']<256*1024**2
